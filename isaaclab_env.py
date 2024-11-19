def make_sim_env(task, room_idx):
    from omni.isaac.lab.app import AppLauncher

    # launch omniverse app
    app_launcher = AppLauncher(enable_cameras=True, headless=False)
    simulation_app = app_launcher.app

    """Rest everything follows."""
    import gymnasium as gym
    import humanoid.tasks
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    
    # parse configuration
    env_cfg = parse_env_cfg(
        task, num_envs=1, use_fabric=True
    )
    env_cfg.use_ik = False
    env_cfg.room_idx = room_idx
    env_cfg.spawn_background = True
    env_cfg.episode_length_s = 16
    env_cfg.seed = 100
    env_cfg.randomize = True
    # create environment
    env = gym.make(task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    return env