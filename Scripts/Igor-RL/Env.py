import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) # DRL_Genesis/Scripts/Igor-RL
parent_dir_1 = os.path.dirname(script_dir) # DRL_Genesis/Scripts
parent_dir_2 = os.path.dirname(parent_dir_1) # DRL_Genesis

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class IgorEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda",capture=False):
        self.device = torch.device(device)
        self.capture = capture
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )


        if self.capture:
            self.cam = self.scene.add_camera(
                # res=(640, 480),
                res=(1920, 1080),
                pos=(2.0, 1.0, 1.5),
                lookat=(0.0, 0.0, 0.5),
                up=(0.0, 0.0, 1.0),
                fov=40,
            aperture=0.0,
            focus_dist=1.0,
            GUI=False,
            spp=1,
            denoise=False,
            )



        # add plain
        # self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)) # デフォルトのplane.urdfを使用
        self.scene.add_entity(gs.morphs.URDF(file=f"{parent_dir_2}/Description/plane/plane.urdf", fixed=True)) # カスタムしたplane.urdfを使用
       

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # self.robot = self.scene.add_entity(
        #     gs.morphs.URDF(
        #         file="path/to/urdf_file",
        #         pos=self.base_init_pos.cpu().numpy(),
        #         quat=self.base_init_quat.cpu().numpy(),
        #     ),
        # )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/mnt/ssd1/ryosei/master/Genesis_WS/Description/Igor/world.xml",
                # pos=self.base_init_pos.cpu().numpy(),
                # quat=self.base_init_quat.cpu().numpy(),
            ),
        )


        # build
        self.scene.build(n_envs=num_envs,env_spacing=(1,1))

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        # self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in ["L_wheel_joint","R_wheel_joint","L_kfe_joint","R_kfe_joint","L_hfe_joint","R_hfe_joint"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * 4, self.motor_dofs[2:])
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * 4, self.motor_dofs[2:])

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_vel = exec_actions[:,0:2] * torch.tensor(self.env_cfg["action_scale"][0:2], device=self.device)
        target_dof_pos = exec_actions[:,2:] * torch.tensor(self.env_cfg["action_scale"][2:], device=self.device) + self.default_dof_pos[2:]
        # self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.robot.control_dofs_velocity(target_dof_vel, self.motor_dofs[0:2])
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs[2:])
        
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 6
                self.dof_vel * self.obs_scales["dof_vel"],  # 6
                self.actions,  # 6
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.capture:
            self.cam.set_pose(
                pos=(self.base_pos[0, 0].cpu().numpy() + 2.0, self.base_pos[0, 1].cpu().numpy() + 1.0, self.base_pos[0, 2].cpu().numpy() + 0.5),
            lookat=(self.base_pos[0, 0].cpu().numpy(), self.base_pos[0, 1].cpu().numpy(), self.base_pos[0, 2].cpu().numpy()-0.3),
                up=(0.0, 0.0, 1.0),
            )
            self.cam.render()
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_leg_energy(self):
        # Penalize leg energy
        return torch.sum(torch.abs(self.dof_vel[:,2:]*self.dof_force[:,2:]), dim=1)

if __name__ == "__main__":
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
            "kp": 50.0,
            "kd": 2.0,
            # termination
            "termination_if_roll_greater_than": 360,  # degree
            "termination_if_pitch_greater_than": 360,
            # base pose
            "base_init_pos": [0.0, 0.0, 0.77],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 20.0,
            "resampling_time_s": 4.0,
            "action_scale": [8.8, 8.8, 1.0, 1.0, 1.0, 1.0], # ["L_wheel_joint (rad/s)","R_wheel_joint (rad/s)","L_kfe_joint (rad)","R_kfe_joint (rad)","L_hfe_joint (rad)","R_hfe_joint (rad)"]
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
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.2,
                "lin_vel_z": -1.0,
                "base_height": -50.0,
                "action_rate": -0.005,
            },
        }
        command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.0, 0.0],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
        }


        return env_cfg, obs_cfg, reward_cfg, command_cfg

    
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    gs.init()
    
    env = IgorEnv(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, device="cuda:0",capture=True)

    obs, _ = env.reset()
    with torch.no_grad():
        # while True:
        env.cam.start_recording()
        for i in range(200):
            actions = torch.zeros(1,6,device="cuda:0")
            actions[0,0] = 1.0
            actions[0,1] = 1.0
            actions[0,2:6] = 0.0
            obs, _, rews, dones, infos = env.step(actions)
            # print(env.r,env.theta,env.phi,env.mu,env.omega,env.psi)
        env.cam.stop_recording(save_to_filename=f"{script_dir}/env_video.mp4", fps=60)