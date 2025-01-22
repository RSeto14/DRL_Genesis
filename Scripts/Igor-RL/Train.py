import argparse
import os
import shutil
import json

from Env import IgorEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import pytz
import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "L_wheel_joint": 0.0,
            "R_wheel_joint": 0.0,
            "L_kfe_joint": -1.0765679639548074,
            "R_kfe_joint": -1.0765679639548074,
            "L_hfe_joint": 0.5,
            "R_hfe_joint": 0.5,
        },
        "dof_names": [
            "L_wheel_joint",
            "R_wheel_joint",
            "L_kfe_joint",
            "R_kfe_joint",
            "L_hfe_joint",
            "R_hfe_joint",
        ],
        # PD
        "kp": 30.0,
        "kd": 2.0,
        # termination
        "termination_if_roll_greater_than": 60,  # degree
        "termination_if_pitch_greater_than": 60,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.77],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": [8.8, 8.8, 0.5, 0.5, 0.5, 0.5], # ["L_wheel_joint (rad/s)","R_wheel_joint (rad/s)","L_kfe_joint (rad)","R_kfe_joint (rad)","L_hfe_joint (rad)","R_hfe_joint (rad)"]
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
            "num_obs": 27,
            "obs_scales": {
                "lin_vel": 1.0,
                "dof_pos": 1.0,
                "dof_vel": 1.0,
                "ang_vel": 1.0,
            },
        }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.7660865336331426,
        "reward_scales": {
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 3.0,
            "lin_vel_z": -1.0,
            # "base_height": -50.0,
            # "action_rate": -0.005,
            # "base_height": -1.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "leg_energy": -0.001,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.5, 0.5],
    }


    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Igor-Turn")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # Device
    visible_device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

    device = "cuda:0"

    gs.init(logging_level="warning")
    
    timezone = pytz.timezone('Asia/Tokyo')
    start_datetime = datetime.datetime.now(timezone)    
    start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    
    log_dir = f"{script_dir}/Logs/{start_formatted}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = IgorEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False, device=device
    )

    runner = OnPolicyRunner(env, train_cfg, f"{log_dir}/Networks", device=device)

    # 複数の設定データをまとめる
    config_data = {
        "train_cfg": train_cfg,
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg
    }

    # JSONファイルに保存
    with open(f'{log_dir}/config.json', 'w') as json_file:
        json.dump(config_data, json_file, indent=4)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
