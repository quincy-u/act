import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import cv2
import h5py

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

left_hand_joint_ids = [26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48]
right_hand_joint_ids = [31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49]

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    is_sim = True if task_name.startswith('Humanoid') else False
    # if is_sim:
    if True:
        if not is_eval:
            from constants import SIM_TASK_CONFIGS
            # if task_name in SIM_TASK_CONFIGS:
            #     task_config = SIM_TASK_CONFIGS[task_name]
            # else:
            f= h5py.File('/home/quincy/dev/act/raw_data/episode_0.hdf5', 'r')
            episode_len = f['observations/qpos'].shape[0]
            task_config = {
                'dataset_dir': '/home/quincy/dev/act/data/',
                'num_episodes': 40,
                'episode_len': episode_len,
                'camera_names': ['main'],
            },
            dataset_dir = '/home/quincy/dev/act/data/'
            num_episodes = 40
            episode_len = episode_len
            camera_names = ['main']
        else:
            episode_len = 450
            num_episodes = 40
            camera_names = ['main']
            task_config = {
                'dataset_dir': '/home/quincy/dev/act/data/',
                'num_episodes': 40,
                'episode_len': episode_len,
                'camera_names': ['main'],
            },
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    # print(task_config)
    # dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    # episode_len = task_config['episode_len']
    # camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 38
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'room_idx': args['room_idx']
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = ['policy_epoch_6000_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def get_image(obs):
    image = obs['fixed_rgb'].squeeze().cpu().numpy()
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)
    curr_image = torch.tensor(rearrange(image, 'h w c -> c h w'))
    curr_image = (curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'main'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from isaaclab_env import make_sim_env
        env = make_sim_env(task_name, config['room_idx'])
        # env_max_reward = env.task.max_reward
    
    from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from omni.isaac.lab.utils.math import subtract_frame_transforms
    # IK controllers
    command_type = "pose"
    left_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="dls")
    left_ik_controller = DifferentialIKController(left_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    right_ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method="pinv")
    right_ik_controller = DifferentialIKController(right_ik_cfg, num_envs=env.scene.num_envs, device=env.sim.device)
    left_jacobin_idx = env.left_ee_idx-1
    right_jacobin_idx = env.right_ee_idx-1

    # Create buffers to store actions
    left_ik_commands_world = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    left_ik_commands_robot = torch.zeros(env.scene.num_envs, left_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_world = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)
    right_ik_commands_robot = torch.zeros(env.scene.num_envs, right_ik_controller.action_dim, device=env.robot.device)

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    max_timesteps = 450

    num_rollouts = 5
    episode_returns = []
    highest_rewards = []
    num_success = 0
    curr_rollout_success = False
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        # ### set task
        # if 'sim_transfer_cube' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        # elif 'sim_insertion' in task_name:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        curr_rollout_success = False

        # ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        #     plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, 50)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        # rewards = []
        with torch.inference_mode():
            print('Env Resetting ....')
            obs, _ = env.reset()
            left_ik_controller.reset()
            right_ik_controller.reset()
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                # if onscreen_render:
                #     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                #     plt_img.set_data(image)
                #     plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                # obs = ts.observation
                # if 'image' in obs:
                #     image_list.append(obs['image'])
                # else:
                #     image_list.append({'main': obs['image']})
                image_list.append({'main': obs['fixed_rgb'].squeeze().cpu().numpy()})
                qpos_numpy = np.array(obs['qpos'].squeeze().cpu().numpy())
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs)
                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = torch.tensor(action, device=env.device)

                # obtain quantities from simulation
                robot_pose_w = env.robot.data.root_state_w[:, 0:7]
                left_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, left_jacobin_idx, :, env.cfg.left_arm_cfg.joint_ids]
                left_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.left_arm_cfg.body_ids[0], 0:7]
                left_joint_pos = env.robot.data.joint_pos[:, env.cfg.left_arm_cfg.joint_ids]
                right_arm_jacobian = env.robot.root_physx_view.get_jacobians()[:, right_jacobin_idx, :, env.cfg.right_arm_cfg.joint_ids]
                right_ee_curr_pose_world = env.robot.data.body_state_w[:, env.cfg.right_arm_cfg.body_ids[0], 0:7]
                right_joint_pos = env.robot.data.joint_pos[:, env.cfg.right_arm_cfg.joint_ids]
                # prepare IK 
                left_ee_curr_pose_robot, left_ee_curr_quat_robot = subtract_frame_transforms(
                    robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ee_curr_pose_world[:, 0:3], left_ee_curr_pose_world[:, 3:7]
                )
                right_ee_curr_pos_robot, right_ee_curr_quat_robot = subtract_frame_transforms(
                    robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ee_curr_pose_world[:, 0:3], right_ee_curr_pose_world[:, 3:7]
                )
                left_ik_commands_world[:, 0:7] = torch.tensor(action[0 : 7], device=env.device)
                left_ik_commands_robot[:, 0:3], left_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
                    robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], left_ik_commands_world[:, 0:3], left_ik_commands_world[:, 3:7]
                )
                right_ik_commands_world[:, 0:7] = torch.tensor(action[7 : 14], device=env.device)
                right_ik_commands_robot[:, 0:3], right_ik_commands_robot[:, 3:7] = subtract_frame_transforms(
                    robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], right_ik_commands_world[:, 0:3], right_ik_commands_world[:, 3:7]
                )
                left_ik_controller.set_command(left_ik_commands_robot, left_ee_curr_pose_robot, left_ee_curr_quat_robot)
                right_ik_controller.set_command(right_ik_commands_robot, right_ee_curr_pos_robot, right_ee_curr_quat_robot)
                # compute the joint commands
                left_joint_pos_des = left_ik_controller.compute(left_ee_curr_pose_robot, left_ee_curr_quat_robot, left_arm_jacobian, left_joint_pos)
                right_joint_pos_des = right_ik_controller.compute(right_ee_curr_pos_robot, right_ee_curr_quat_robot, right_arm_jacobian, right_joint_pos)
            
                target_qpos = torch.zeros(size=(1, 50), device=env.device)
                target_qpos[:, env.cfg.left_arm_cfg.joint_ids] = left_joint_pos_des
                target_qpos[:, env.cfg.right_arm_cfg.joint_ids] = right_joint_pos_des
                target_qpos[:, left_hand_joint_ids] = action[14:26]
                target_qpos[:, right_hand_joint_ids] = action[26:38]

                ### step the environment
                obs, _, _, _, _ = env.step(target_qpos)
                if obs['success'] and not curr_rollout_success:
                    num_success += 1
                    curr_rollout_success = True
                    break

                # ### for visualization
                # qpos_list.append(qpos_numpy)
                # target_qpos_list.append(target_qpos)
                # rewards.append(ts.reward)

            # plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        # rewards = np.array(rewards)
        # episode_return = np.sum(rewards[rewards!=None])
        # episode_returns.append(episode_return)
        # episode_highest_reward = np.max(rewards)
        # highest_rewards.append(episode_highest_reward)
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        print(f'{num_success} successful rollouts out of {num_rollouts}.')
        if save_episode:
            room_idx = config['room_idx']
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{room_idx}{rollout_id}.mp4'))

    # success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    # avg_return = np.mean(episode_returns)
    # summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    # for r in range(env_max_reward+1):
    #     more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    #     more_or_equal_r_rate = more_or_equal_r / num_rollouts
    #     summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    # print(summary_str)

    # save success rate to txt
    # result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    result_file_name = 'eval_summary.txt'
    with open(os.path.join('/home/quincy/dev/act/', result_file_name), 'a') as f:
        f.write(task_name + ', ' + str(config['room_idx']) + ': ' + str(num_success))
        f.write('\n')
        f.close()

    return 0, 0


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # isaaclab
    parser.add_argument('--room_idx', action='store', type=int, help='room_idx', required=False, default=1)
    
    main(vars(parser.parse_args()))
