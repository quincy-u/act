import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# STATE_NAMES = JOINT_NAMES + ["gripper"]
STATE_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
left_hand_joint_ids = [26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48]
right_hand_joint_ids = [31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49]

def resize_images_bilinear(batch_images, target_width, target_height):
    # Extract the trajectory length and original dimensions
    trajectory_length, original_height, original_width, channels = batch_images.shape

    # Prepare an array to hold the resized images
    resized_images = np.zeros((trajectory_length, target_height, target_width, channels), dtype=np.uint8)

    # Resize each image in the batch
    for i in range(trajectory_length):
        resized_images[i] = cv2.resize(batch_images[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized_images


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if '/observations/left_ee_pose' not in root:
            left_ee_pose = root['/observations/left_curr_ee_pose'][()]
            right_ee_pose = root['/observations/right_curr_ee_pose'][()]
        else:
            left_ee_pose = root['/observations/left_ee_pose'][()]
            right_ee_pose = root['/observations/right_ee_pose'][()]
        left_finger_pos = root['/action'][()][:, left_hand_joint_ids]
        right_finger_pos = root['/action'][()][:, right_hand_joint_ids]
        action = np.concatenate([left_ee_pose, right_ee_pose, left_finger_pos, right_finger_pos], axis=-1)
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            if cam_name == 'main':
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        root.close()

    return qpos, qvel, action, image_dict,action.shape[0] 

def pad_hdf5(dataset_dir, target_dir, dataset_name, total_length):
    input_dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    output_dataset_path = os.path.join(target_dir, dataset_name + '.hdf5')

    with h5py.File(input_dataset_path, 'r') as input_file, h5py.File(output_dataset_path, 'w') as output_file:
        images = input_file['/observations/images/main'][()]
        images = resize_images_bilinear(images, 384, 384)
        qpos = input_file['/observations/qpos'][()]
        qvel = input_file['/observations/qvel'][()]
        if '/observations/left_ee_pose' not in input_file:
            left_ee_pose = input_file['/observations/left_curr_ee_pose'][()]
            right_ee_pose = input_file['/observations/right_curr_ee_pose'][()]
        else:
            left_ee_pose = input_file['/observations/left_ee_pose'][()]
            right_ee_pose = input_file['/observations/right_ee_pose'][()]
        left_finger_pos = input_file['/action'][()][:, left_hand_joint_ids]
        right_finger_pos = input_file['/action'][()][:, right_hand_joint_ids]
        action = np.concatenate([left_ee_pose, right_ee_pose, left_finger_pos, right_finger_pos], axis=-1)
        # Set attributes and create datasets in the new file
        output_file.attrs['sim'] = input_file.attrs['sim']
        # Create datasets in the new file
        output_file.create_dataset('/observations/images/main', (total_length,) + images.shape[1:], dtype=images.dtype)
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
        input_file.close()
        output_file.close()


        

def main(args):
    dataset_dir = args['dataset_dir']
    target_dir = args['target_dir']
    num_episode = args['num_episode']
    
    max_episode_len = 0
    for episode_idx in range(num_episode):
        dataset_name = f'episode_{episode_idx}'
        qpos, qvel, action, image_dict, episode_len = load_hdf5(dataset_dir, dataset_name)
        max_episode_len = max(episode_len, max_episode_len)
    print(f'max episode len: {max_episode_len}')
    for episode_idx in range(num_episode):
        dataset_name = f'episode_{episode_idx}'
        pad_hdf5(dataset_dir, target_dir, dataset_name, max_episode_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--target_dir', action='store', type=str, help='Target dir.', required=True)
    parser.add_argument('--num_episode', action='store', type=int, help='Number of episodes.', required=False)
    main(vars(parser.parse_args()))
