import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Use Wandb to monitor training
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Navigation Training Environment
    from env import NavigationEnv
    #TODO 创建一个无人机导航环境，定义了无人机的状态、动作和奖励函数
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = [] # 用于对观测和动作进行预处理或后处理的 转换列表
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    # 位置控制器：根据 目标位置 计算 无人机的控制指令
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)

    # 速度转换器类：位置控制 ==> 速度控制，以便无人机能够执行动作
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)

    # 转换后的环境：将 原始环境 和 速度转换器 组合，用于强化学习训练。.train() 将环境设置为训练模式
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    

    #TODO PPO Policy：训练无人机的导航策略
    policy = PPO(cfg.algo,
                 transformed_env.observation_spec,  # 环境的观测空间，即无人机可以观察到的状态
                 transformed_env.action_spec,   # 环境的动作空间，即无人机可以执行的动作
                 cfg.device)

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/xinmingh/RLDrones/navigation/scripts/nav-ros/navigation_runner/ckpts/checkpoint_36000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector，初始化 数据收集器和统计器
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)  # 收集和统计每个episode（一轮训练）的数据，例如奖励、步数等

    # RL Data Collector，从 env 中收集数据，并将数据传递给强化学习算法
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,    # 每个batch的数据帧数 = 机器人个数 * 训练帧数
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace 时序差分值 (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # 随机探索策略 sample from normal distribution
    )

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Inf，记录训练信息，例如环境帧数和每秒帧数
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # Train Policy
        train_loss_stats = policy.train(data)   # 使用 从 env 中收集的数据 训练 强化学习策略，并获取训练损失信息
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        episode_stats.add(data) # 将 从 env 中收集的数据 添加到 一个episode统计器中
        if len(episode_stats) >= transformed_env.num_envs: # 环境中的每个agent都训练了一个episode，则进行评测
            # 计算一个episode统计的信息的均值，如 平均奖励，平均步数
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # Evaluate policy and log info，到达 评测 帧数，则进行评估
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            env.enable_render(True) # 启用渲染，以便可视化无人机的导航过程
            env.eval()  # 环境设置为评估模式
            # 使用评估策略评估无人机的导航性能
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            )
            env.enable_render(not cfg.headless)
            env.train() # 转回训练模式
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # Update wand info
        run.log(info)


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    