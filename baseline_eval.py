import gc

from click import File
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

# Scope for full access to the files in Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive']
left_hand_joint_ids = [26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48]
right_hand_joint_ids = [31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49]

def list_subfolders(path):
    # Set to hold the names of subfolders
    subfolder_names = set()
    
    # os.walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for root, dirs, files in os.walk(path):
        for name in dirs:
            subfolder_names.add(name)
    
    return list(subfolder_names)  # Convert the set to a list for the function output

def main():
    ckpt_home_dir = '/home/quincy/dev/act/ckpt'
    tasks = list_subfolders(ckpt_home_dir)
    tasks.sort()
    for task_name_shorten in tqdm(tasks):
        if task_name_shorten  in ['Stack-Single-Cube', 'Unload-Cans' ]:
            task_name = f'Humanoid-{task_name_shorten}-v0'
            with open('/home/quincy/dev/act/eval_summary.txt', 'a') as f:
                f.write(f'\n===============================================================================================\n')
                f.close()
            for room_id in range(3):
                print('='*100)
                room_idx = room_id +1
                print(f'Processing task: {task_name}, room: {room_idx}')

                ckpt_dir = f'/home/quincy/dev/act/ckpt/{task_name_shorten}'
                if not os.path.exists(ckpt_dir):
                    raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist.")
                imitate_episodes_args = ['--task_name', task_name, '--policy_class', 'ACT', '--kl_weight' ,'10' ,'--chunk_size' ,'150' ,'--hidden_dim' ,'512', 
                                        '--batch_size', '64' ,'--dim_feedforward', '3200' ,'--num_epochs', '2000',  '--lr' ,'5e-5' ,'--seed', '0' ,'--ckpt_dir' ,
                                        ckpt_dir, '--room_idx', str(room_idx), '--eval']
                imitate_episodes_script_path = '/home/quincy/dev/act/imitate_episodes.py'
                result = subprocess.run(['python', imitate_episodes_script_path] + imitate_episodes_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("Output:", result.stdout)
                print("Errors:", result.stderr)
                        
                gc.collect()
if __name__ == '__main__':
    main()