import argparse
import gc

import numpy as np
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.http import build_http
from google_auth_httplib2 import AuthorizedHttp
import pickle
import os
from tqdm import tqdm
import h5py
import io
import subprocess
import httplib2
import cv2

# Scope for full access to the files in Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive']
left_hand_joint_ids = [26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48]
right_hand_joint_ids = [31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49]

def authenticate_google_drive():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            # pip install google-api-python-client==1.7.2 google-auth==1.8.0 google-auth-httplib2==0.0.3 google-auth-oauthlib==0.4.1
            creds = flow.run_console()
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            
    http = httplib2.Http(timeout=300)  # Timeout in seconds
    authorized_http = AuthorizedHttp(creds, http=http)
    return build('drive', 'v3', http=authorized_http)

def list_files(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        spaces='drive',
        fields='nextPageToken, files(id, name, mimeType)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    return results.get('files', [])

def find_file(service, folder_id, file_name):
    # Search for specific file by name within the folder
    query = f"name = '{file_name}' and '{folder_id}' in parents"
    response = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = response.get('files', [])
    return files[0] if files else None

def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def upload_file(service, file_name, parent_id, memory_file):
    print(f"Uploading {file_name}...    ")
    file_metadata = {
        'name': file_name,
        'parents': [parent_id],
    }
    media = MediaIoBaseUpload(memory_file, mimetype='application/octet-stream')
    # Note the additional parameter `supportsAllDrives=True`
    file = service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
    return file.get('id')

def create_folder(service, folder_name, parent_id):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(
        body=file_metadata,
        fields='id',
        supportsAllDrives=True,
    ).execute()
    return folder.get('id')

def pad_hdf5(source_hdf5, target_dir, file_name, total_length):
    output_dataset_path = os.path.join(target_dir, file_name)

    with  h5py.File(output_dataset_path, 'w') as output_file:
        images = source_hdf5['/observations/images/main'][()]
        qpos = source_hdf5['/observations/qpos'][()]
        qvel = source_hdf5['/observations/qvel'][()]
        left_ee_pose = source_hdf5['/observations/left_ee_pose'][()]
        right_ee_pose = source_hdf5['/observations/right_ee_pose'][()]
        left_finger_pos = source_hdf5['/action'][()][:, left_hand_joint_ids]
        right_finger_pos = source_hdf5['/action'][()][:, right_hand_joint_ids]
        action = np.concatenate([left_ee_pose, right_ee_pose, left_finger_pos, right_finger_pos], axis=-1)

        # Set attributes and create datasets in the new file
        output_file.attrs['sim'] = source_hdf5.attrs['sim']

        # Create datasets in the new file
        output_file.create_dataset('/observations/images/main', (total_length,) + source_hdf5['/observations/images/main'].shape[1:], dtype=input_file['/observations/images/main'].dtype)
        output_file.create_dataset('/observations/qpos', (total_length, qpos.shape[1]), dtype=qpos.dtype)
        output_file.create_dataset('/observations/qvel', (total_length, qvel.shape[1]), dtype=qvel.dtype)
        output_file.create_dataset('/observations/left_ee_pose', (total_length, left_ee_pose.shape[1]), dtype=left_ee_pose.dtype)
        output_file.create_dataset('/observations/right_ee_pose', (total_length, right_ee_pose.shape[1]), dtype=right_ee_pose.dtype)
        output_file.create_dataset('/action', (total_length, action.shape[1]), dtype=action.dtype)

        # Write the original data
        output_file['/observations/images/main'][:qpos.shape[0], :] = images
        output_file['/observations/qpos'][:qpos.shape[0], :] = qpos
        output_file['/observations/qvel'][:qvel.shape[0], :] = qvel
        output_file['/observations/left_ee_pose'][:left_ee_pose.shape[0], :] = left_ee_pose
        output_file['/observations/right_ee_pose'][:right_ee_pose.shape[0], :] = right_ee_pose
        output_file['/action'][:action.shape[0], :] = action

        len_to_pad = total_length - qpos.shape[0]
        # Pad the data
        if len_to_pad > 0:
            output_file['/observations/qpos'][-len_to_pad:, :] = qpos[-1]
            output_file['/observations/qvel'][-len_to_pad:, :] = qvel[-1]
            output_file['/observations/left_ee_pose'][-len_to_pad:, :] = left_ee_pose[-1]
            output_file['/observations/right_ee_pose'][-len_to_pad:, :] = right_ee_pose[-1]
            output_file['/action'][-len_to_pad:, :] = action[-1]
            output_file['/observations/images/main'][-len_to_pad:, :] = images[-1]
        source_hdf5.close()
        print('Saving to ', output_dataset_path)
        output_file.close()

def main(parent_dir):
    folder_link = 'https://drive.google.com/drive/folders/1I2QcdQqSNcgVDoDkO0AmPwfz3TkCIUhp'  # Replace with your Google Drive folder link
    folder_id = folder_link.split('/')[-1]  # Extract ID from the link
    service = authenticate_google_drive()
    tasks = list_files(service, folder_id)
    tasks.sort(key=lambda x: x['name'])
    task_ls = []
    for task in tqdm(tasks, desc='Tasks'):
        if task['mimeType'] == 'application/vnd.google-apps.folder' and task['name'] in ['Orient-Pour-Balls']: 
            task_name_shorten = task['name']
            task_ls.append(task_name_shorten)
            task_name = f'Humanoid-{task_name_shorten}-v0'
            print('='*100)
            print(f'Processing task: {task_name}')
            episode_idx = 0
            file_name = f"episode_{episode_idx}.hdf5"
            file = find_file(service, task['id'], file_name)
            if not file:
                print(f"File {file_name} not found in {task_name_shorten}")
            else:
                print(f"Downloading {file_name} from folder {task_name_shorten}")
                file = download_file(service, file['id'])
                with h5py.File(file, 'r') as source_hdf5:
                    image = source_hdf5['/observations/images/main'][30]
                    image = image[:, :, [2, 1, 0]]  # swap B and R channel
                    cv2.imwrite(f'/home/quincy/dev/act/imgs/Sim-{task_name_shorten}.png', image)
            gc.collect()
    print(task_ls)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    args = parser.parse_args()
    parent_dir = '/data/quincyu' if args.server else '/home/quincy/dev'
    main(parent_dir)