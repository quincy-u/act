def make_sim_env(task):
    from omni.isaac.lab.app import AppLauncher

    # launch omniverse app
    app_launcher = AppLauncher(enable_cameras=True)
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
    # create environment
    env = gym.make(task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    return env