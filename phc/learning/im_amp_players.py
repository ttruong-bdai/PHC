import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs

COLLECT_Z = False

# Takara
sys.path.insert(0,'/home/ttruong/sim_suite')

from diffusion_policy import DIFFUSION_POLICY_ROOT
from diffusion_policy.trainer.base_trainer import BaseTrainer

import dill
import hydra 
import collections
import torch.nn.functional as F
from phc.utils import torch_utils
def load_diffusion_policy(checkpoint_path, tag='latest'):    
    # payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    payload = BaseTrainer.load_wandb_checkpoint(run_path=checkpoint_path, tag=tag)

    cfg = payload["cfg"]
    
    # OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    
    trainer: BaseTrainer = cls(cfg)
    trainer.load_payload(payload, exclude_keys=None, include_keys=None)

    import importlib.util

    # Path to the file and the class name
    data_class_target = trainer.cfg.dataset._target_
    parts = data_class_target.split('.')

    # Extract the module path and class name
    module_path = '.'.join(parts[:-1])   
    class_name = parts[-1]   
    module = importlib.import_module(module_path)

    # Get the class from the module
    dataset_class = getattr(module, class_name)
    # import ipdb; ipdb.set_trace()

    policy = trainer.agent
    if cfg.training.use_ema:
        policy = trainer.ema_agent
    
    policy.to('cuda')
    policy.eval() 
    return policy, cfg, dataset_class 


# Takara
# def load_diffusion_policy(checkpoint_path):
#     payload = BaseTrainer.load_wandb_checkpoint(checkpoint_path, tag='latest')
#     # payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
#     cfg = payload["cfg"]
#     # OmegaConf.resolve(cfg)
#     cls = hydra.utils.get_class(cfg._target_)
#     trainer: BaseTrainer = cls(cfg)
#     trainer.load_payload(payload, exclude_keys=None, include_keys=None)
#     policy = trainer.agent
#     if cfg.training.use_ema:
#         policy = trainer.ema_agent
#     policy.to('cuda')
#     policy.eval()
#     return policy, cfg, trainer


class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 0.5 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0
        
        # Takara 
        self.motion_lib = humanoid_env._motion_lib 
        self.mode = config['mode']
        self.act_noise = config['act_noise']
        self.collect_start_idx = config['collect_start_idx']
        self.collect_step_idx = config['collect_step_idx'] if config['collect_step_idx'] else self.env.num_envs
        self.collect_end_idx = self.collect_start_idx + self.collect_step_idx

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return
    
    def action_rotation_6d_to_euler(self,act_rot6d):
        # act_rot6d = torch_utils.matrix_to_rotation_6d(torch_utils.euler_angles_to_matrix(act.reshape(self.env.num_envs, -1,3) ,'XYZ')).reshape(self.env.num_envs, -1)  
        action_converted = torch_utils.matrix_to_euler_angles(torch_utils.rotation_6d_to_matrix(act_rot6d.reshape(act_rot6d.shape[0], -1, 6)),'XYZ').reshape(act_rot6d.shape[0], -1)
        return action_converted 
    

    # verify it works here 
    def global_to_characterFrame(self, obs):    
        init_root_pos = obs[:,0,0:3].clone() 
        init_root_rot = obs[:,0,3:7].clone()  # first frame
        
        body_pos = obs[:,:,7: 7+72].reshape((obs.shape[0], obs.shape[1], -1 ,3))  #(num_envs, num_horizon_steps, num_bodies, pos_xyz)
        body_vel = obs[:,:,79:].reshape((obs.shape[0], obs.shape[1], -1, 3)) 
        
        # import ipdb; ipdb.set_trace()
        E, H, J, _ = body_pos.shape 

        heading_inv_rot = torch_utils.calc_heading_quat_inv(init_root_rot)

        # center pos obs  
        body_pos[:,:,:,:2] -= init_root_pos[:,None,None,:].repeat(1, H, J, 1)[:,:,:,:2]
        # rotate points 

        local_pos = torch_utils.my_quat_rotate(heading_inv_rot[:,None,None,:].repeat(1,H,J,1).view(-1,4), body_pos.reshape(-1,3)).reshape(E,H,J,-1)  

        # ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        local_vel = torch_utils.my_quat_rotate(heading_inv_rot[:,None,None,:].repeat(1,H,J,1).view(-1,4), body_vel.reshape(-1,3)).reshape(E,H,J,-1)  
        obs_characterFrame = torch.cat((local_pos, local_vel),dim=-1).view(E,H,-1) 
        # import ipdb; ipdb.set_trace()

        # noise=  torch.zeros((E,H,69))
        # obs_characterFrame = torch.cat((obs_characterFrame, noise),dim=-1) 

        # obs_characterFrame = local_pos.view(E,H,-1)  

        return obs_characterFrame

        
    def run(self):
        is_determenistic = self.is_determenistic
        render = self.render_env
        need_init_rnn = self.is_rnn
        obs_dict = self.env_reset()
        
        batch_size = self.get_batch_size(obs_dict["obs"], 1)
        steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        max_steps =  int(self.motion_lib.get_motion_num_steps().max().cpu().numpy())
        done_envs = np.zeros(self.env.num_envs, dtype=bool)
        term_envs = np.zeros(self.env.num_envs, dtype=bool)
        done_indices = []        

        if self.mode == 'diff':
            
            # checkpoint_path = '/home/ttruong/sim_suite/outputs/November-13-10-48-29-autoregressive_historyDiffused_studentforced_myObs/checkpoints/latest.ckpt'
            # checkpoint_path = '/home/ttruong/sim_suite/outputs/November-13-14-15-43-coddiffuse_EE_obs/checkpoints/latest.ckpt'
            #v1
            checkpoint_path = 'bdaii/diffusion_policy_codiffuse/m5zxoocr'
            
            #v2
            checkpoint_path ='bdaii/diffusion_policy_codiffuse/470nbx3k'
            
            #v3
            checkpoint_path = 'bdaii/diffusion_policy_codiffuse/43pvzzk6'

            #v4
            checkpoint_path = 'bdaii/diffusion_policy_codiffuse/lsr0o8xx'

            # robust v1
            checkpoint_path = 'bdaii/diffusion_policy_codiffuse/oh9g8k4t'

            # robust v4 
            checkpoint_path = 'bdaii/diffusion_policy_codiffuse/0c4qomm5'
            policy, dp_cfg, dataset_class = load_diffusion_policy(checkpoint_path,'950')

            obs_deque = collections.deque([self.env.task.phc_obs] * dp_cfg.policy.actor.backbone.n_obs_steps, maxlen = dp_cfg.policy.actor.backbone.n_obs_steps)
        
        if self.mode == 'collect':
            if flags.rand_start:
                max_steps = 100 
                collect_window = 20
                self.env.task._shift_character()
        

            self.env.task.use_noisy_action = True 
            self.env.task.act_noise = self.act_noise
            obs_store = np.zeros((self.env.num_envs, max_steps, 319+2*69)) # 216 phc obs 
            act_store = np.zeros((self.env.num_envs, max_steps, 69))
            act_rot6d_store = np.zeros((self.env.num_envs, max_steps, 23*6)) #138 

            act_noisy_store = np.zeros((self.env.num_envs, max_steps, 69))
            act_rot6d_noisy_store = np.zeros((self.env.num_envs, max_steps, 23*6)) #138 
                    
        n=0
        with torch.no_grad():
            while n < max_steps:
                obs_dict = self.env_reset(done_indices)
                observation = self.env.task.phc_obs.clone() # This is for recording 

                action = self.get_action(obs_dict, is_determenistic) 

                if self.mode=='diff':
                    
                    # dataset_class._state_normalize()
                    global_stacked_obs = torch.tensor(np.stack(list(obs_deque))).permute((1,0,2))
                    # diff_obs = diff_obs[:,:,7:]

                    root_pos = global_stacked_obs[:,:,0:3]
                    root_rot = global_stacked_obs[:,:,3:7]
                    # import ipdb
                    B, H = root_rot.shape[:2]
                    J = 24 

                    body_pos = global_stacked_obs[:,:,7: 7+72].view(B, H, J, 3).clone()
                    body_rot = global_stacked_obs[:,:, 79: 79+96].view(B, H, J, 4).clone()
                
                    body_lin_vel = global_stacked_obs[:,:, 175: 175+72].view(B, H, J, 3).clone()
                    body_ang_vel = global_stacked_obs[:,:, 247: 247+72].view(B, H, J, 3).clone()
                    
                    joint_pos = global_stacked_obs[:,:, 319:319+69].view(B, H, -1, 3).clone()
                    joint_vel = global_stacked_obs[:,:, 388: 388+69].view(B, H, -1, 3).clone()
                    
                    # # import ipdb; ipdb.set_trace() 
                    # diff_obs = dataset_class._state_normalize(root_pos, root_rot, body_pos, body_rot, body_lin_vel,body_ang_vel, 3) # 3 hardcoded 
                    handsfeet_idx = np.array([ 3,  7, 18, 23])# self.getbodyidx({'L_Ankle', 'R_Ankle', 'L_Hand', 'R_Hand'})

                    # diff_obs = dataset_class._state_normalize(root_pos, root_rot, body_pos, body_rot, body_lin_vel,body_ang_vel, handsfeet_idx) # 3 hardcoded 
                    diff_obs = dataset_class._state_normalize(root_pos, root_rot, body_pos, body_rot, body_lin_vel,body_ang_vel, 3, handsfeet_idx) # 3 hardcoded 

                    # diff_obs = dataset_class._state_normalize(root_pos, root_rot, body_pos, body_rot, body_lin_vel, 3) # 3 hardcoded 
                    # diff_obs = dataset_class._state_normalize( root_pos, root_rot, body_pos, body_lin_vel, 0) # 3 hardcoded 

                    # diff_obs  = self.global_to_characterFrame(diff_obs)
                    # import ipdb; ipdb.set_trace()
                    action = policy.act({'obs':diff_obs})
                    action= action[:,3,:] 
                    # import ipdb; ipdb.set_trace()
                    # action = actio
                    # action= action[:,3,:69] 
                    
                    # action = self.action_rotation_6d_to_euler(action)

                # Step Environment
                obs_dict, r, done, info = self.env_step(self.env, action)

                # import ipdb; ipdb.set_trace()
                if self.mode=='diff':
                    # obs_deque.append(self.env.task.phc_obs.clone( ))
                    obs_deque.append(self.env.task.phc_obs.clone( ))

                    # add eval flag
                    
                if self.mode == 'collect':
                    try:
                        obs_store[~done_envs,n,:] = observation[~done_envs,:].clone()
                        
                        act = self.env.task.mean_action.reshape(1,-1) if self.env.num_envs ==1 else self.env.task.mean_action
                        act_store[~done_envs,n,:] = act[~done_envs,:].clone() 

                        act_noisy = self.env.task.noisy_action.reshape(1,-1) if self.env.num_envs ==1 else self.env.task.noisy_action
                        act_noisy_store[~done_envs,n,:] = act_noisy[~done_envs,:].clone() 

                        # rot6d conversion 
                        act_rot6d_store[~done_envs,n,:] =torch_utils.matrix_to_rotation_6d(torch_utils.euler_angles_to_matrix(act.reshape(self.env.num_envs, -1,3),'XYZ')).reshape(self.env.num_envs, -1)[~done_envs,:]
                        act_rot6d_noisy_store[~done_envs,n,:] =torch_utils.matrix_to_rotation_6d(torch_utils.euler_angles_to_matrix(act_noisy.reshape(self.env.num_envs, -1,3),'XYZ')).reshape(self.env.num_envs, -1)[~done_envs,:]
                        
                        # action_converted = self.action_rotation_6d_to_euler(act)
                        # assert torch.isclose(act, action_converted).all()                         
                        n+=1 
                        
                    except:
                        print('error in collection')
                        import ipdb; ipdb.set_trace()

                ep_lens = (self.env.task._motion_lib.get_motion_num_steps() - 1 ).cpu().numpy()
                
                done_envs = n >= ep_lens 
                term_envs = term_envs | (self.env.task._terminate_buf.cpu().numpy() & ~done_envs).astype(bool)   

                if render:
                    self.env.render(mode="human")
                    time.sleep(self.render_sleep)

        if self.mode == 'collect':
            failed_names = None
            if term_envs.any():
                failed_idx = torch.tensor(term_envs).nonzero(as_tuple=False).squeeze()
                failed_names = self.motion_lib.curr_motion_keys[failed_idx]
                print(f'failed motions:{failed_names}')

            data_dir = f'rollout_data/'
            if flags.rand_start: 
                data_fname=  f'rand_start_rollout-collect_{collect_window}-{max_steps}_motions_{self.collect_start_idx}-{self.collect_end_idx}_noise_{self.act_noise}_numSucc_{ self.env.num_envs - term_envs.sum()}_numTerm_{term_envs.sum()}'
                clip_ep_len = collect_window
            else:
                data_fname=  f'full_rollout-motions_{self.collect_start_idx}-{self.collect_end_idx}_noise_{self.act_noise}_numSucc_{ self.env.num_envs - term_envs.sum()}_numTerm_{term_envs.sum()}'
                clip_ep_len = max_steps

            data_path = os.path.join(data_dir, data_fname) 

            # Save the Data (REMOVE TERMINATED EPISODES)

            ep_names = self.motion_lib.curr_motion_keys[~term_envs]
            ep_lens = self.motion_lib.get_motion_num_steps().cpu().numpy()[~term_envs] -1 
            
            if flags.rand_start: 
                ep_lens[:] = clip_ep_len
            
            obs = obs_store[~term_envs,:clip_ep_len]
        
            act = act_store[~term_envs,:clip_ep_len] 
            act_rot6d = act_rot6d_store[~term_envs,:clip_ep_len]
            act_noisy = act_noisy_store[~term_envs,:clip_ep_len] 
            act_rot6d_noisy = act_rot6d_noisy_store[~term_envs,:clip_ep_len]


            # deconstruct global observation here 
            root_pos = obs[:,:,0:3]
            root_rot = obs[:,:,3:7]
            
            body_pos = obs[:,:,7: 7+72]
            body_rot = obs[:,:, 79: 79+96]

            body_lin_vel = obs[:,:, 175: 175+72]
            body_ang_vel = obs[:,:, 247: 247+72]
            
            joint_pos = obs[:,:, 319:319+69]
            joint_vel = obs[:,:, 388: 388+69]

            np.savez(
                data_path,
                obs=obs,
                root_pos = root_pos, 
                root_rot = root_rot, 
                body_pos = body_pos, 
                body_rot = body_rot, 
                body_lin_vel = body_lin_vel,
                body_ang_vel = body_ang_vel, 
                
                joint_pos = joint_pos, 
                joint_vel = joint_vel,

                act=act, 
                act_rot6d= act_rot6d,
                act_noisy=act_noisy, 
                act_rot6d_noisy= act_rot6d_noisy,

                ep_len = ep_lens, 
                ep_name = ep_names,
                failed_ep_names = failed_names,
            )
            
            print(f'collection done saved to {data_path}')
        # import ipdb; ipdb.set_trace()

        return  





    # def _post_step(self, info, done):
    #     super()._post_step(info)
        
        
    #     # Takara
    #     max_steps = self.env.task._motion_lib.get_motion_num_steps() -1 
    #     done = self.curr_stpes >= max_steps 
    #     self.curr_stpes += 1

    #     return done 

    #     # modify done such that games will exit and reset.
    #     if flags.im_eval:

    #         humanoid_env = self.env.task
            
    #         termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
    #         # termination_state = info["terminate"]
    #         self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
    #         if (~self.terminate_state).sum() > 0:
    #             max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
    #             curr_ids = humanoid_env._motion_lib._curr_motion_ids
    #             if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
    #                 bound = (max_possible_id == curr_ids).nonzero()[0] + 1
    #                 if (~self.terminate_state[:bound]).sum() > 0:
    #                     curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
    #                 else:
    #                     curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
    #             else:
    #                 curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

    #             if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
    #         else:
    #             curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

    #         self.mpjpe.append(info["mpjpe"])
    #         self.gt_pos.append(info["body_pos_gt"])
    #         self.pred_pos.append(info["body_pos"])
    #         if COLLECT_Z: self.zs.append(info["z"])
    #         self.curr_stpes += 1

    #         if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                
    #             self.terminate_memory.append(self.terminate_state.cpu().numpy())
    #             self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

    #             # MPJPE
    #             all_mpjpe = torch.stack(self.mpjpe)
    #             try:
    #                 assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
    #             except:
    #                 import ipdb; ipdb.set_trace()
    #                 print('??')

    #             all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
    #             all_body_pos_pred = np.stack(self.pred_pos)
    #             all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
    #             all_body_pos_gt = np.stack(self.gt_pos)
    #             all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

    #             if COLLECT_Z:
    #                 all_zs = torch.stack(self.zs)
    #                 all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
    #                 self.zs_all += all_zs


    #             self.mpjpe_all.append(all_mpjpe)
    #             self.pred_pos_all += all_body_pos_pred
    #             self.gt_pos_all += all_body_pos_gt
                

    #             if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
    #                 terminate_hist = np.concatenate(self.terminate_memory)
    #                 succ_idxes = np.nonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()

    #                 pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
    #                 gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

    #                 pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
    #                 gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

    #                 # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
    #                 # humanoid_env._motion_lib.get_motion_num_steps().sum()

    #                 failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
    #                 success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
    #                 # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
    #                 if flags.real_traj:
    #                     pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
    #                     gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
    #                     pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
    #                     gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
    #                 metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
    #                 metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

    #                 metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
    #                 metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

    #                 print("------------------------------------------")
    #                 print("------------------------------------------")
    #                 print(f"Success Rate: {self.success_rate:.10f}")
    #                 print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
    #                 print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
    #                 # print(1 - self.terminate_state.sum() / self.terminate_state.shape[0])
    #                 print(self.config['network_path'])
    #                 if COLLECT_Z:
    #                     zs_all = self.zs_all[:humanoid_env._motion_lib._num_unique_motions]
    #                     zs_dump = {k: zs_all[idx].cpu().numpy() for idx, k in enumerate(humanoid_env._motion_lib._motion_data_keys)}
    #                     joblib.dump(zs_dump, osp.join(self.config['network_path'], "zs_run.pkl"))
                    
    #                 import ipdb; ipdb.set_trace()

    #                 # joblib.dump(np.concatenate(self.zs_all[: humanoid_env._motion_lib._num_unique_motions]), osp.join(self.config['network_path'], "zs.pkl"))

    #                 joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
    #                 joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
    #                 print("....")

    #             done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

    #             humanoid_env.forward_motion_samples()
    #             self.terminate_state = torch.zeros(
    #                 self.env.task.num_envs, device=self.device
    #             )

    #             self.pbar.update(1)
    #             self.pbar.refresh()
    #             self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
    #             if COLLECT_Z: self.zs = []
    #             self.curr_stpes = 0


    #         update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
    #         self.pbar.set_description(update_str)


    #     return done
    
    # def get_z(self, obs_dict):
    #     obs = obs_dict['obs']
    #     if self.has_batch_dimension == False:
    #         obs = unsqueeze_obs(obs)
    #     obs = self._preproc_obs(obs)
    #     input_dict = {
    #         'is_train': False,
    #         'prev_actions': None,
    #         'obs': obs,
    #         'rnn_states': self.states
    #     }
    #     with torch.no_grad():
    #         z = self.model.a2c_network.eval_z(input_dict)
    #         return z


#     def run(self):
#         n_games = self.games_num
#         render = self.render_env
#         n_game_life = self.n_game_life
#         is_determenistic = self.is_determenistic
#         sum_rewards = 0
#         sum_steps = 0
#         sum_game_res = 0
#         n_games = n_games * n_game_life
#         games_played = 0
#         has_masks = False
#         has_masks_func = getattr(self.env, "has_action_mask", None) is not None

#         op_agent = getattr(self.env, "create_agent", None)
#         if op_agent:
#             agent_inited = True

#         if has_masks_func:
#             has_masks = self.env.has_action_mask()

#         need_init_rnn = self.is_rnn
#         for t in range(n_games):
#             if games_played >= n_games:
#                 break
#             obs_dict = self.env_reset()

#             batch_size = 1
#             batch_size = self.get_batch_size(obs_dict["obs"], batch_size)
#             if need_init_rnn:
#                 self.init_rnn()
#                 need_init_rnn = False
#             # cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
#             steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
#             # print_game_res = False
#             done_indices = []


#             # Takara
#             max_steps = self.motion_lib.get_motion_num_steps().max() 
#             ep_lens = self.motion_lib.get_motion_num_steps()
#             obs_store = np.zeros((self.env.num_envs, max_steps, 216))
#             act_store = np.zeros((self.env.num_envs, max_steps, 69))
#             done_envs = np.zeros(self.env.num_envs, dtype=bool)
            

#             if self.mode == 'collect':
#                 se

#             with torch.no_grad():
#                 for n in range(self.max_steps):
#                     print(n)
#                     # Takara, this prevents sim from returning only the done
#                     # if self.mode =='collect':
#                     #     obs_dict = self.env_reset(done_indices)

#                     obs_dict = self.env_reset(done_indices)
#                     observation = self.env.task.phc_obs.clone() # This is for recording 

#                     action = self.get_action(obs_dict, is_determenistic)

#                     # Step Environment
#                     obs_dict, r, done, info = self.env_step(self.env, action)

#                     if self.mode == 'collect':
#                         try:
#                             obs_store[~done_envs,n,:] = observation[~done_envs,:].clone()
#                             act_store[~done_envs,n,:] = self.env.task.mean_action.reshape(1,-1)[~done_envs,:].clone() if self.env.num_envs ==1 else self.env.task.mean_action[~done_envs,:].clone()  
#                         except:
#                             import ipdb; ipdb.set_trace()

#                     if done.any():
#                         import ipdb; ipdb.set_trace()
#                     # done = self._post_step(info, done.clone())
#                     max_steps = self.env.task._motion_lib.get_motion_num_steps() - 1 
                    
#                     done_envs = steps >= max_steps 
#                     term_envs = self.env.task._terminate_buf.clone()  

#                     steps[term_envs] = 0 

#                     if render:
#                         self.env.render(mode="human")
#                         time.sleep(self.render_sleep)
                    
#                     # all_done_indices = done.nonzero(as_tuple=False)
#                     # done_indices = all_done_indices[:: self.num_agents]
#                     # done_count = len(done_indices)
# # 
#                     # TAKARA
#                     # done_envs[done_indices] = True
#                     # print(done_envs) 
# # 
#                     # done_indices = done_indices[:, 0]
                    
#                     steps +=1 

#                     if done_envs.all():
#                         print('done')
#                         import ipdb; ipdb.set_trace() 


"""

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def quats_to_rot_matrices(quats):
    squeeze_flag = False
    if quats.dim() == 1:
        squeeze_flag = True
        quats = torch.unsqueeze(quats, 0)
    nq = torch.linalg.vecdot(quats, quats, dim=1)
    singularities = nq < 1e-10
    result = torch.zeros(quats.shape[0], 3, 3, device=quats.device)
    result[singularities] = torch.eye(3, device=quats.device).reshape((1, 3, 3)).repeat(sum(singularities), 1, 1)
    non_singular = quats[torch.logical_not(singularities)] * torch.sqrt(2.0 / nq).reshape((-1, 1)).repeat(1, 4)
    non_singular = torch.einsum("bi,bj->bij", non_singular, non_singular)
    result[torch.logical_not(singularities), 0, 0] = 1.0 - non_singular[:, 2, 2] - non_singular[:, 3, 3]
    result[torch.logical_not(singularities), 0, 1] = non_singular[:, 1, 2] - non_singular[:, 3, 0]
    result[torch.logical_not(singularities), 0, 2] = non_singular[:, 1, 3] + non_singular[:, 2, 0]
    result[torch.logical_not(singularities), 1, 0] = non_singular[:, 1, 2] + non_singular[:, 3, 0]
    result[torch.logical_not(singularities), 1, 1] = 1.0 - non_singular[:, 1, 1] - non_singular[:, 3, 3]
    result[torch.logical_not(singularities), 1, 2] = non_singular[:, 2, 3] - non_singular[:, 1, 0]
    result[torch.logical_not(singularities), 2, 0] = non_singular[:, 1, 3] - non_singular[:, 2, 0]
    result[torch.logical_not(singularities), 2, 1] = non_singular[:, 2, 3] + non_singular[:, 1, 0]
    result[torch.logical_not(singularities), 2, 2] = 1.0 - non_singular[:, 1, 1] - non_singular[:, 2, 2]
    if squeeze_flag:
        result = torch.squeeze(result)
    return result

@torch.jit.script
def matrices_to_euler_angles(mat, extrinsic: bool = True):
    _POLE_LIMIT = 1.0 - 1e-6
    if extrinsic:
        north_pole = mat[:, 2, 0] > _POLE_LIMIT
        south_pole = mat[:, 2, 0] < -_POLE_LIMIT
        result = torch.zeros(mat.shape[0], 3, device=mat.device)
        result[north_pole, 0] = 0.0
        result[north_pole, 1] = -np.pi / 2
        result[north_pole, 2] = torch.arctan2(mat[north_pole, 0, 1], mat[north_pole, 0, 2])
        result[south_pole, 0] = 0.0
        result[south_pole, 1] = np.pi / 2
        result[south_pole, 2] = torch.arctan2(mat[south_pole, 0, 1], mat[south_pole, 0, 2])
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0] = torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 1],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 2],
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1] = -torch.arcsin(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 0]
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2] = torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1, 0],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 0],
        )
    else:
        north_pole = mat[:, 2, 0] > _POLE_LIMIT
        south_pole = mat[:, 2, 0] < -_POLE_LIMIT
        result = torch.zeros(mat.shape[0], 3, device=mat.device)
        result[north_pole, 0] = torch.arctan2(mat[north_pole, 1, 0], mat[north_pole, 1, 1])
        result[north_pole, 1] = np.pi / 2
        result[north_pole, 2] = 0.0
        result[south_pole, 0] = torch.arctan2(mat[south_pole, 1, 0], mat[south_pole, 1, 1])
        result[south_pole, 1] = -np.pi / 2
        result[south_pole, 2] = 0.0
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0] = -torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1, 2],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2, 2],
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 1] = torch.arcsin(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 2]
        )
        result[torch.logical_not(torch.logical_or(south_pole, north_pole)), 2] = -torch.arctan2(
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 1],
            mat[torch.logical_not(torch.logical_or(south_pole, north_pole)), 0, 0],
        )
    return result



@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw, extrinsic: bool = True):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    if extrinsic:
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
    else:
        qw = -sr * sp * sy + cr * cp * cy
        qx = sr * cp * cy + sp * sy * cr
        qy = -sr * sy * cp + sp * cr * cy
        qz = sr * sp * cy + sy * cr * cp
    return torch.stack([qw, qx, qy, qz], dim=-1)

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([w, x, y, z], dim=-1).view(shape)
    return quat

@torch.jit.script
def get_euler_xyz(q, extrinsic: bool = True):
    if extrinsic:
        qw, qx, qy, qz = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    else:
        result = matrices_to_euler_angles(quats_to_rot_matrices(q), extrinsic=False)
        return result[:, 0], result[:, 1], result[:, 2]
@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

    # def rotate_xiaoyu():
        # B, T, N = trajectory.shape[:3]
        # 
# 
        # first_quaternion = trajectory[:,0, 0:4].clone() # data in [x,y,z,w]
        # roll, pitch, yaw  = get_euler_xyz(first_quaternion[:,[3,0,1,2]])  # Extract yaw-pitch-roll 
        # yaw_correction_quat = quat_from_euler_xyz(roll * 0, pitch * 0, -yaw)  # retruns [w,x,y,z]
        # 
    #    quat = trajectory[..., 0:4][[3,0,1,2]].reshape(-1,4).clone()
     #   corrected_quat = quat_mul(yaw_correction_quat.repeat(T,1), quat)
        # import ipdb; ipdb.set_trace() 
# 
        # pos = trajectory[:,:,4: 4+72].reshape(-1,3).clone()
        # local_pos2 = pos.clone()
        # 
        # first_position = trajectory[:,0,4:7]
        # local_pos2[:,:2] -=first_position.repeat_interleave(T*J,1).reshape(-1,3)[:,:2]
# 
        # corrected_pos = quat_rotate(yaw_correction_quat.repeat_interleave(T*J,1).reshape(-1,4), local_pos2)
# 
"""