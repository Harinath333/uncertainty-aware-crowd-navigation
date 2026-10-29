from . import VecEnvWrapper
import numpy as np
from .running_mean_std import RunningMeanStd
import torch
import os
from collections import deque

import copy
import pickle

from gst_updated.src.gumbel_social_transformer.temperature_scheduler import Temp_Scheduler
from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv

class VecPretextNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that processes the observations and rewards, used for GST predictors
    and returns from an environment.
    config: a Config object
    test: whether we are training or testing
    """

    def __init__(self, venv, ob=False, ret=False, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, config=None, test=False):
        VecEnvWrapper.__init__(self, venv)

        self.config = config
        self.device=torch.device(self.config.training.device)
        if test:
            self.num_envs = 1
        else:
            self.num_envs = self.config.env.num_processes

        self.max_human_num = config.sim.human_num + config.sim.human_num_range

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = torch.zeros(self.num_envs).to(self.device)
        self.gamma = gamma
        self.epsilon = epsilon

        # load and configure the prediction model
        load_path = os.path.join(os.getcwd(), self.config.pred.model_dir)
        if not os.path.isdir(load_path):
            raise RuntimeError('The result directory was not found.')
        checkpoint_dir = os.path.join(load_path, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)

        self.predictor = CrowdNavPredInterfaceMultiEnv(load_path=load_path, device=self.device, config = self.args, num_env = self.num_envs)

        temperature_scheduler = Temp_Scheduler(self.args.num_epochs, self.args.init_temp, self.args.init_temp, temp_min=0.03)
        self.tau = temperature_scheduler.decay_whole_process(epoch=100)

        # handle different prediction and control frequency
        self.pred_interval = int(self.config.data.pred_timestep//self.config.env.time_step)
        self.buffer_len = (self.args.obs_seq_len - 1) * self.pred_interval + 1



    def talk2Env_async(self, data):
        self.venv.talk2Env_async(data)


    def talk2Env_wait(self):
        outs=self.venv.talk2Env_wait()
        return outs

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # process the observations and reward
        obs, rews = self.process_obs_rew(obs, done, rews=rews)

        return obs, rews, done, infos

    def _obfilt(self, obs):
        if self.ob_rms and self.config.RLTrain:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        # queue for inputs to the pred model
        # fill the queue with dummy values
        self.traj_buffer = deque(list(-torch.ones((self.buffer_len, self.num_envs, self.max_human_num, 2), device=self.device)*999),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        self.mask_buffer = deque(list(torch.zeros((self.buffer_len, self.num_envs, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)

        self.step_counter = 0

        # for calculating the displacement of human positions
        self.last_pos = torch.zeros(self.num_envs, self.max_human_num, 2).to(self.device)

        obs = self.venv.reset()
        obs, _ = self.process_obs_rew(obs, np.zeros(self.num_envs))

        return obs


    '''
    1. Process observations: 
    Run inference on pred model with past obs as inputs, fill in the predicted trajectory in O['spatial_edges']
    
    2. Process rewards: 
    Calculate reward for colliding with predicted future traj and add to the original reward, 
    same as calc_reward() function in crowd_sim_pred.py except the data are torch tensors
    '''
    def process_obs_rew(self, O, done, rews=0.):
        # O:
        # robot_node: [nenv, 1, 7]
        # spatial_edges: [nenv, max_human_num, 2*(1+predict_steps)]
        # visible_masks: [nenv, max_human_num]

        # ============================================================
        # PREPARE HUMAN POSITIONS
        # ============================================================

        # Human absolute positions in world frame
        human_pos = (
            O['robot_node'][:, :, :2]
            + O['spatial_edges'][:, :, :2]
        )

        # Insert new observation into trajectory buffer
        self.traj_buffer.append(human_pos)
        self.mask_buffer.append(
            O['visible_masks'].unsqueeze(-1)
        )

        # [obs_seq_len, nenv, humans, 2]
        # ->
        # [nenv, humans, obs_seq_len, 2]
        in_traj = torch.stack(
            list(self.traj_buffer)
        ).permute(1, 2, 0, 3)

        in_mask = torch.stack(
            list(self.mask_buffer)
        ).permute(1, 2, 0, 3).float()

        # ============================================================
        # PREPARE GST INPUT
        # ============================================================

        in_traj = in_traj[:, :, ::self.pred_interval]
        in_mask = in_mask[:, :, ::self.pred_interval]

        # ============================================================
        # GST PREDICTION
        # ============================================================

        out_traj, out_mask = self.predictor.forward(
            input_traj=in_traj,
            input_binary_mask=in_mask
        )

        # ============================================================
        # GET MULTIPLE GST HYPOTHESES
        # ============================================================

        self.multiple_out_traj = getattr(
            self.predictor,
            "last_multiple_samples",
            None
        )

        # ============================================================
        # UNCERTAINTY ESTIMATION
        # ============================================================

        if self.multiple_out_traj is not None:

            multiple_samples = self.multiple_out_traj

            # Ensure tensor
            if not torch.is_tensor(multiple_samples):
                multiple_samples = torch.as_tensor(
                    multiple_samples,
                    device=self.device,
                    dtype=torch.float32
                )

            # Shape:
            #
            # [H, N, T, P, 2]
            #
            # H = hypotheses
            # N = environments
            # T = prediction steps
            # P = humans
            # 2 = x,y

            # --------------------------------------------------------
            # Mean trajectory
            # --------------------------------------------------------

            self.pred_mean = multiple_samples.mean(dim=0)

            # --------------------------------------------------------
            # Variance across hypotheses
            # --------------------------------------------------------

            self.pred_variance = multiple_samples.var(
                dim=0,
                unbiased=False
            )

            # --------------------------------------------------------
            # Standard deviation in x/y
            # --------------------------------------------------------

            self.pred_std = torch.sqrt(
                self.pred_variance + 1e-6
            )

            # --------------------------------------------------------
            # Scalar uncertainty magnitude
            # --------------------------------------------------------

            self.pred_uncertainty = torch.sqrt(
                self.pred_variance[..., 0]
                + self.pred_variance[..., 1]
            )

            # ========================================================
            # ATTENTION CIRCLE
            # ========================================================

            # Robot position for each environment
            robot_positions = O['robot_node'][:, 0, :2]

            # Human positions
            human_positions = human_pos

            # Attention radius
            attention_radius = getattr(
                self.config.env,
                "circle_radius",
                getattr(
                    self.config.robot,
                    "sensor_range",
                    6.0
                )
            )

            # Distance from robot to every human
            human_distances = torch.linalg.norm(
                human_positions - robot_positions.unsqueeze(1),
                dim=-1
            )

            # Humans inside attention circle
            attention_mask = (
                human_distances <= attention_radius
            )

            # Also require human to actually be visible/valid
            attention_mask = torch.logical_and(
                attention_mask,
                O['visible_masks'].bool()
            )

            # ========================================================
            # COMMON REPORT VARIABLES
            # ========================================================

            num_hypotheses = multiple_samples.shape[0]
            num_envs = multiple_samples.shape[1]
            num_steps = multiple_samples.shape[2]
            num_humans = multiple_samples.shape[3]

            # Environment 0 is used for the readable test output
            env_id = 0

            visible_humans = torch.where(
                attention_mask[env_id]
            )[0].tolist()


            # ========================================================
            # PROBABILISTIC INTRUSION RISK
            # ========================================================

            # multiple_samples:
            # [H, N, T, P, 2]

            # Robot position expanded to:
            # [1, N, 1, 1, 2]

            robot_positions_expanded = (
                robot_positions
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(2)
            )

            # Current human WORLD positions: [N, P, 2]
            # -> [1, N, 1, P, 2]
            human_positions_expanded = (
                human_positions.unsqueeze(0)
                .unsqueeze(2)
            )

            # GST multi-samples are relative to each human's current position.
            # Convert them to WORLD coordinates.
            multiple_samples_world = (
                multiple_samples + human_positions_expanded
            )

            # Distance from every predicted human position to robot
            predicted_distances = torch.linalg.norm(
                multiple_samples_world - robot_positions_expanded,
                dim=-1
            )
            # [H, N, T, P]

            # ========================================================
            # INTRUSION RADIUS
            # ========================================================

            intrusion_radius = (
                self.config.robot.radius
                + self.config.humans.radius
            )

            # ========================================================
            # CHECK EACH HYPOTHESIS
            # ========================================================

            intrusion_events = (
                predicted_distances <= intrusion_radius
            )

            # ========================================================
            # INTRUSION PROBABILITY
            # ========================================================

            # Number of hypotheses entering intrusion region
            # divided by total number of hypotheses

            self.intrusion_probability = (
                intrusion_events.float()
                .mean(dim=0)
            )

            # Shape:
            # [N, T, P]

            # Only keep humans inside attention circle
            self.intrusion_probability = (
                self.intrusion_probability
                * attention_mask.unsqueeze(1).float()
            )

            # ========================================================
            # MAXIMUM INTRUSION PROBABILITY
            # ========================================================

            self.max_intrusion_probability = (
                self.intrusion_probability.max(dim=1).values
            )


        # ============================================================
        # EXISTING DETERMINISTIC COLLISION REWARD
        # ============================================================

        out_mask = out_mask.bool()

        hr_dist_future = (
            out_traj[:, :, :, :2]
            - O['robot_node'][:, :, :2].unsqueeze(1)
        )

        collision_idx = (
            torch.norm(
                hr_dist_future,
                dim=-1
            )
            <
            self.config.robot.radius
            + self.config.humans.radius
        )

        # Mask invalid predictions
        collision_idx = torch.logical_and(
            collision_idx,
            out_mask
        )

        coefficients = (
            2. ** torch.arange(
                2,
                self.config.sim.predict_steps + 2,
                device=self.device
            ).reshape(
                (1, 1, self.config.sim.predict_steps)
            )
        )

        collision_penalties = (
            self.config.reward.collision_penalty
            / coefficients
        )

        reward_future = (
            collision_idx.to(torch.float)
            * collision_penalties
        )

        reward_future, _ = torch.min(
            reward_future.reshape(
                self.num_envs,
                -1
            ),
            dim=1
        )

        # ============================================================
        # PROBABILISTIC GST INTRUSION RISK REWARD
        # ============================================================

        if hasattr(self, "max_intrusion_probability"):
            # Maximum predicted intrusion probability among
            # humans inside the robot attention circle.
            #
            # Shape:
            #   max_intrusion_probability = [N, P]
            #   risk_score = [N]

            risk_score = self.max_intrusion_probability.max(
                dim=1
            ).values

            # Use the normal collision penalty as the scale.
            # collision_penalty is negative, therefore this
            # automatically produces a negative reward penalty.
            risk_penalty = (
                self.config.reward.collision_penalty
                * risk_score
            )

        else:
            # GST risk is unavailable during initialization/fallback.
            risk_score = torch.zeros(
                self.num_envs,
                device=self.device
            )

            risk_penalty = torch.zeros(
                self.num_envs,
                device=self.device
            )

        # Add both:
        #   1. deterministic predicted collision reward
        #   2. probabilistic GST intrusion-risk penalty

        total_reward = (
            reward_future
            + risk_penalty
        )

        # Compact GST debug — print once every 1000 steps
        if (self.step_counter + 1) % 1000 == 0:
            print(
                f"[GST] step={self.step_counter} "
                f"risk={risk_score.mean().item():.2f} "
                f"penalty={risk_penalty.mean().item():.2f} "
                f"reward={rews.mean():.2f}"
            )

        rews = (
            rews
            + total_reward.reshape(
                self.num_envs,
                1
            ).cpu().numpy()
        )

        # ============================================================
        # CONVERT PREDICTION BACK TO ROBOT FRAME
        # ============================================================

        robot_pos = (
            O['robot_node'][:, :, :2]
            .unsqueeze(1)
        )

        out_traj[:, :, :, :2] = (
            out_traj[:, :, :, :2]
            - robot_pos
        )

        out_mask = out_mask.repeat(
            1,
            1,
            self.config.sim.predict_steps * 2
        )

        new_spatial_edges = (
            out_traj[:, :, :, :2]
            .reshape(
                self.num_envs,
                self.max_human_num,
                -1
            )
        )

        O['spatial_edges'][:, :, 2:][out_mask] = (
            new_spatial_edges[out_mask]
        )

        # ============================================================
        # SORT HUMANS BY DISTANCE
        # ============================================================

        hr_dist_cur = torch.linalg.norm(
            O['spatial_edges'][:, :, :2],
            dim=-1
        )

        sorted_idx = torch.argsort(
            hr_dist_cur,
            dim=1
        )

        for i in range(self.num_envs):
            O['spatial_edges'][i] = (
                O['spatial_edges'][i][sorted_idx[i]]
            )

        # ============================================================
        # FINAL OBSERVATION
        # ============================================================

        obs = {
            'robot_node': O['robot_node'],
            'spatial_edges': O['spatial_edges'],
            'temporal_edges': O['temporal_edges'],
            'visible_masks': O['visible_masks'],
            'detected_human_num': O['detected_human_num'],
        }

        self.last_pos = copy.deepcopy(human_pos)
        self.step_counter += 1

        return obs, rews
