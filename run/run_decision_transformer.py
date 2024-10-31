import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from torch.serialization import load
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle
import torch
import os
import torch.multiprocessing as mp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

model_path = "./saved_model/DTtest/"

def run_dt():
    train_model("./data/trajectory/autoBidding_aigb_track_final_data_trajectory_data_1.csv")
    train_model("./data/trajectory/autoBidding_aigb_track_final_data_trajectory_data_2.csv", load_from=True)
    train_model("./data/trajectory/autoBidding_aigb_track_final_data_trajectory_data_3.csv", load_from=True)

def train_model(path, load_from=False, save=True, model_name="dt.pt", select_range=(0, 47), K=48, max_ep_len=48, step_num=30000):
    state_dim = 16
    logger.info(model_name)
    logger.info(f"Start processing {path}, range {select_range}")
    replay_buffer = EpisodeReplayBuffer(state_dim, 1, path, select_range=select_range, K=K, max_ep_len=max_ep_len)

    logger.info(f"Replay buffer size: {len(replay_buffer.sorted_inds)}")
    
    # Initialize the model here but do not push it to the device yet
    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std, K=K, max_ep_len=max_ep_len)


    
    batch_size = 32
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size, num_workers=8, pin_memory=True)

    # Now push the model to the device (CUDA) after dataloader is created
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_from:
        model.load_net(load_path=model_path+model_name, device=device)
    model.to(device)
    model.train()
    i = 0
    for batch in dataloader:
        states, actions, rewards, dones, rtg, timesteps, attention_mask = [x.to(device) for x in batch]
        
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        i += 1
        if i % 1000 == 0:
            logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
        model.scheduler.step()

    if save == True:
        model.save_net(model_path, model_name=model_name) 
    logger.info("Saved")


def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_dt()
