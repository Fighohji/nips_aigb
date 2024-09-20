import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import pickle
import random
import os

class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path, max_ep_len=48, scale=2000, K=24):
        self.device = "cpu"
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)

        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(ValueError)
                return val

        training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
        self.trajectories = training_data

        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones, self.realCPA, self.constraintCPA = [], [], [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        for index, row in self.trajectories.iterrows():
            state.append(row["state"])
            reward.append(row['reward'])
            action.append(row["action"])
            dones.append(row["done"])
            if row["done"]:
                if len(state) != 1:
                    self.states.append(np.array(state))
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    self.realCPA.append(max(float(row["realAllCost"]) / max(float(row["realAllConversion"]), 1e-4), 1e-4))
                    self.constraintCPA.append(max(1e-4, float(row["CPAConstraint"])))
                state = []
                reward = []
                action = []
                dones = []
        self.traj_lens, self.returns, self.constraintCPA, self.realCPA = np.array(self.traj_lens), np.array(self.returns), np.array(self.constraintCPA), np.array(self.realCPA)

        picklePath = os.path.join("saved_model", "DTtest", "normalize_dict.pkl")

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)

        self.state_mean, self.state_std = np.array(normalize_dict["state_mean"]), np.array(normalize_dict["state_std"])  # 求出每一个状态特征的均值标准差

        self.trajectories = []
        for i in range(len(self.states)):
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i]})

        ''''''
        self.K = K
        self.pct_traj = 0.6
        ''''''

        # together
        ratio_squared = (self.constraintCPA / self.realCPA) ** 2

        # 使用 np.maximum 保证比值平方的最小值为 1
        scale_factor = np.minimum(ratio_squared, 1)

        # 计算 1 / (1 + exp(-10 * ((traj_lens - 0.5)))) 的结果，逐元素执行
        tot_lens = np.sum(self.traj_lens)
        p_sample_base = 1 / (1 + np.exp(-10 * ((self.traj_lens / tot_lens - 0.5))))

        # 逐元素乘以 rewards
        score = p_sample_base * scale_factor * self.returns

        sorted_inds = np.argsort(score)  # lowest to highest
        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0:
            if timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
                timesteps += self.traj_lens[sorted_inds[ind]]
                num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]
        score = score[self.sorted_inds]
        
        self.p_sample = score / sum(score)


    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, traj['rewards'].shape[0] - 1)

        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, rtg, timesteps, mask

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __len__(self):
        return len(self.sorted_inds)