"""
MuZero 訓練腳本
完全獨立於現有的 train.py
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

from RL import agent, buffer, environment, model
from Structure import structure
from Validation import check_design, design_space
from Visualization import plot, visualize
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_muzero_args() -> argparse.Namespace:
    """解析 MuZero 專用的命令行參數"""
    parser = argparse.ArgumentParser()
    
    # 基本設定
    parser.add_argument("--ckpt_dir", type=Path, default="./Results/MuZero/")
    parser.add_argument("--random_seed", type=int, default=731)
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or auto")
    
    # 模型參數
    parser.add_argument("--model_type", type=str, default="Taiwan", help="Taiwan, Japan")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)
    
    # MuZero 參數
    parser.add_argument("--muzero_num_simulations", type=int, default=50)
    parser.add_argument("--muzero_unroll_steps", type=int, default=3)
    parser.add_argument("--muzero_temperature", type=float, default=1.0)
    parser.add_argument("--muzero_discount", type=float, default=0.99)
    
    # 訓練參數
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_capacity", type=int, default=1000)
    parser.add_argument("--training_frequency", type=int, default=10)
    parser.add_argument("--evaluation_frequency", type=int, default=20)
    parser.add_argument("--save_frequency", type=int, default=20)
    
    # 環境參數
    parser.add_argument("--structure_shape", type=str, default="fixed")
    parser.add_argument("--add_structure_geometry", action="store_true", default=True)
    parser.add_argument("--add_response_features", action="store_true", default=True)
    parser.add_argument("--reward_type", type=str, default="material")
    parser.add_argument("--restrict_action", action="store_true", default=False)
    parser.add_argument("--scwb_driven_design", action="store_true", default=False)
    
    # 非線性動態分析
    parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
    parser.add_argument("--check_acceleration", action="store_true", default=False)
    parser.add_argument("--check_displacement", action="store_true", default=True)
    parser.add_argument("--graph_lstm_dir", type=Path, default=None)
    parser.add_argument("--ground_motion_dir", type=Path, default=None)
    parser.add_argument("--ground_motion_number", type=int, default=11)
    
    args = parser.parse_args()
    return args


def set_random_seed(seed: int):
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)


def get_device(device_arg: str) -> str:
    """取得計算設備"""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def setup_logging(ckpt_dir: Path):
    """設定日誌"""
    logger = logging.getLogger('MuZero-RL')
    logger.setLevel(logging.INFO)
    
    # 設定格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台處理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 檔案處理器
    file_handler = logging.FileHandler(ckpt_dir / "muzero_training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def calculate_n_step_return(game: buffer.MuZeroGame, start_index: int, 
                          unroll_steps: int, discount: float) -> float:
    """
    計算 N-step bootstrapped return
    
    Args:
        game: 遊戲軌跡
        start_index: 起始索引
        unroll_steps: 展開步數
        discount: 折扣因子
    
    Returns:
        n_step_return: N-step return 值
    """
    n_step_return = 0.0
    discount_factor = 1.0
    
    # 計算實際獎勵部分
    for i in range(unroll_steps):
        step_index = start_index + i
        if step_index >= len(game.rewards):
            break
        n_step_return += discount_factor * game.rewards[step_index]
        discount_factor *= discount
    
    # 如果還有更多步，使用網路預測的價值進行 bootstrapping
    if start_index + unroll_steps < len(game.values):
        n_step_return += discount_factor * game.values[start_index + unroll_steps]
    elif len(game.values) > 0:
        # 使用最後一個價值預測
        n_step_return += discount_factor * game.values[-1]
    
    return n_step_return


def muzero_train_step(network: model.MuZeroNetwork, optimizer: torch.optim.Optimizer,
                     replay_buffer: buffer.MuZeroReplayBuffer, batch_size: int, 
                     unroll_steps: int, discount: float, device: str, logger):
    """ 執行一步 MuZero 訓練 """
    if len(replay_buffer) < batch_size:
        return 0.0
    
    # 從緩衝區採樣遊戲
    games = replay_buffer.sample(batch_size)
    
    total_loss = torch.tensor(0.0, device=device)
    
    for game in games:
        try:
            # 隨機選擇起始時間步
            if len(game) <= unroll_steps:
                start_index = 0
            else:
                start_index = torch.randint(0, len(game) - unroll_steps, (1,)).item()
            
            # 獲取初始觀察
            initial_obs = game.observations[start_index]
            
            # 確保 GraphData 包含必要的屬性
            if not hasattr(initial_obs, 'story_batch'):
                num_edges = initial_obs.edge_attr.size(0) // 2
                initial_obs.story_batch = torch.zeros(num_edges, dtype=torch.long)
            
            # 計算 structure_story_ptr（這裡需要從遊戲數據中重建）
            # 由於我們沒有完整的 structure 物件，我們使用預設值
            if not hasattr(initial_obs, 'structure_story_ptr'):
                initial_obs.structure_story_ptr = None
            
            # 確保觀察在正確的設備上
            initial_obs = initial_obs.to(device)
            
            hidden_state = network.represent(initial_obs)
            
            # 計算當前遊戲的動作數量
            current_num_actions = len(game.policies[start_index])
            
            policy_logits, value_pred = network.predict(hidden_state, current_num_actions)
            
            # 計算真實的 N-step return
            true_value = calculate_n_step_return(game, start_index, unroll_steps, discount)
            
            # 計算損失
            value_loss = torch.nn.functional.mse_loss(
                value_pred.squeeze(), 
                torch.tensor([true_value], dtype=torch.float32, device=device)
            )
            
            # 策略損失 - 使用 KL 散度
            target_policy = game.policies[start_index].to(device)
            policy_probs = torch.softmax(policy_logits.squeeze(), dim=-1)
            policy_loss = torch.nn.functional.kl_div(
                torch.log(policy_probs + 1e-8), 
                target_policy + 1e-8, 
                reduction='batchmean'
            )
            
            loss = value_loss + policy_loss
            total_loss += loss
            
        except Exception as e:
            logger.warning(f"Error in training step: {e}")
            continue
    
    if total_loss > 0:
        # 更新網路
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        return total_loss.item() / len(games)
    else:
        return 0.0


def muzero_self_play(env, muzero_agent: agent.MuZeroAgent, replay_buffer: buffer.MuZeroReplayBuffer, logger):
    """ 執行一場 MuZero 自我對弈 """
    current_game = buffer.MuZeroGame()
    structure = env.reset()  # 環境返回的是 structure 物件
    obs = structure.graph.clone()  # 提取 GraphData 物件並複製
    
    # 確保 GraphData 包含必要的屬性
    if hasattr(structure, 'aux') and 'story_batch' in structure.aux:
        obs.story_batch = structure.aux['story_batch'].clone()
    else:
        # 如果沒有 story_batch，創建一個預設值
        num_edges = obs.edge_attr.size(0) // 2  # 每個 member 有兩條邊
        obs.story_batch = torch.zeros(num_edges, dtype=torch.long)
    
    # 計算 structure_story_ptr
    if hasattr(structure, 'aux'):
        structure_story_ptr = []
        story_count = 0
        structure_story_ptr.append(story_count)
        for story_members in (structure.aux["story_xdir_beam_member"] + 
                            structure.aux["story_zdir_beam_member"] + 
                            structure.aux["story_outer_column_member"] + 
                            structure.aux["story_inner_column_member"]):
            story_count += 1
        structure_story_ptr.append(story_count)
        obs.structure_story_ptr = torch.tensor(structure_story_ptr)
    else:
        obs.structure_story_ptr = None
    
    if not hasattr(obs, 'batch'):
        obs.batch = None
    
    # 確保所有張量都在正確的設備上
    device = muzero_agent.device
    obs = obs.to(device)
    
    done = False
    step_count = 0
    
    logger.info("Starting self-play episode")
    
    while not done and step_count < 1000:  # 限制最大步數避免無限循環
        try:
            # 獲取合法動作
            legal_actions = env.get_legal_actions(structure)
            
            if not legal_actions:
                logger.info("No legal actions available, episode finished")
                break
            
            # 選擇動作（只考慮合法動作）
            action, policy = muzero_agent.select_action(obs, training=True, num_actions=len(legal_actions))
            
            # 將動作索引映射到實際的動作
            actual_action = legal_actions[action]
            
            # 計算當前狀態的價值預測
            with torch.no_grad():
                hidden_state = muzero_agent.network.represent(obs)
                _, value_pred = muzero_agent.network.predict(hidden_state)
            
            # 執行動作
            next_structure, reward, done, _, _ = env.step(structure, actual_action)
            next_obs = next_structure.graph.clone()  # 提取下一個狀態的 GraphData
            
            # 確保下一個觀察也包含必要的屬性
            if hasattr(next_structure, 'aux') and 'story_batch' in next_structure.aux:
                next_obs.story_batch = next_structure.aux['story_batch'].clone()
            else:
                # 如果沒有 story_batch，創建一個預設值
                num_edges = next_obs.edge_attr.size(0) // 2
                next_obs.story_batch = torch.zeros(num_edges, dtype=torch.long)
            
            # 計算下一個狀態的 structure_story_ptr
            if hasattr(next_structure, 'aux'):
                structure_story_ptr = []
                story_count = 0
                structure_story_ptr.append(story_count)
                for story_members in (next_structure.aux["story_xdir_beam_member"] + 
                                    next_structure.aux["story_zdir_beam_member"] + 
                                    next_structure.aux["story_outer_column_member"] + 
                                    next_structure.aux["story_inner_column_member"]):
                    story_count += 1
                structure_story_ptr.append(story_count)
                next_obs.structure_story_ptr = torch.tensor(structure_story_ptr)
            else:
                next_obs.structure_story_ptr = None
            
            if not hasattr(next_obs, 'batch'):
                next_obs.batch = None
            
            # 確保下一個觀察也在正確的設備上
            next_obs = next_obs.to(device)
            
            # 儲存步驟數據（轉回 CPU 以節省記憶體）
            current_game.add_step(obs.to('cpu'), actual_action, reward, policy.to('cpu'), done, value_pred.item())
            
            structure = next_structure
            obs = next_obs
            step_count += 1
            
        except Exception as e:
            logger.warning(f"Error in self-play step {step_count}: {e}")
            break
    
    # 將遊戲加入緩衝區
    if len(current_game) > 0:
        replay_buffer.push(current_game)
        logger.info(f"Self-play episode finished with {len(current_game)} steps, total reward: {sum(current_game.rewards):.3f}")
    
    return len(current_game), sum(current_game.rewards)


def muzero_inference(env, muzero_agent: agent.MuZeroAgent, num_episodes: int = 5, logger=None):
    """ 執行 MuZero inference 評估 """
    inference_results = {
        'episodes': [],
        'total_rewards': [],
        'episode_lengths': [],
        'final_designs': [],
        'material_savings': [],
        'performance_metrics': []
    }
    
    logger.info(f"Starting MuZero inference evaluation with {num_episodes} episodes")
    
    for episode in range(num_episodes):
        structure = env.reset()
        obs = structure.graph.clone()
        
        # 確保 GraphData 包含必要的屬性
        if hasattr(structure, 'aux') and 'story_batch' in structure.aux:
            obs.story_batch = structure.aux['story_batch'].clone()
        else:
            num_edges = obs.edge_attr.size(0) // 2
            obs.story_batch = torch.zeros(num_edges, dtype=torch.long)
        
        # 計算 structure_story_ptr
        if hasattr(structure, 'aux'):
            structure_story_ptr = []
            story_count = 0
            structure_story_ptr.append(story_count)
            for story_members in (structure.aux["story_xdir_beam_member"] + 
                                structure.aux["story_zdir_beam_member"] + 
                                structure.aux["story_outer_column_member"] + 
                                structure.aux["story_inner_column_member"]):
                story_count += 1
            structure_story_ptr.append(story_count)
            obs.structure_story_ptr = torch.tensor(structure_story_ptr)
        else:
            obs.structure_story_ptr = None
        
        if not hasattr(obs, 'batch'):
            obs.batch = None
        
        # 確保所有張量都在正確的設備上
        device = muzero_agent.device
        obs = obs.to(device)
        
        done = False
        step_count = 0
        episode_rewards = []
        episode_actions = []
        episode_material_savings = []
        
        while not done and step_count < 1000:
            # 獲取合法動作
            legal_actions = env.get_legal_actions(structure)
            
            if not legal_actions:
                logger.info(f"Episode {episode}: No legal actions available")
                break
            
            # 使用貪婪策略進行 inference
            action, _ = muzero_agent.select_action(obs, training=False, num_actions=len(legal_actions))
            actual_action = legal_actions[action]
            
            # 執行動作
            next_structure, reward, done, _, _ = env.step(structure, actual_action)
            next_obs = next_structure.graph.clone()
            
            # 確保下一個觀察也包含必要的屬性
            if hasattr(next_structure, 'aux') and 'story_batch' in next_structure.aux:
                next_obs.story_batch = next_structure.aux['story_batch'].clone()
            else:
                num_edges = next_obs.edge_attr.size(0) // 2
                next_obs.story_batch = torch.zeros(num_edges, dtype=torch.long)
            
            # 計算下一個狀態的 structure_story_ptr
            if hasattr(next_structure, 'aux'):
                structure_story_ptr = []
                story_count = 0
                structure_story_ptr.append(story_count)
                for story_members in (next_structure.aux["story_xdir_beam_member"] + 
                                    next_structure.aux["story_zdir_beam_member"] + 
                                    next_structure.aux["story_outer_column_member"] + 
                                    next_structure.aux["story_inner_column_member"]):
                    story_count += 1
                structure_story_ptr.append(story_count)
                next_obs.structure_story_ptr = torch.tensor(structure_story_ptr)
            else:
                next_obs.structure_story_ptr = None
            
            if not hasattr(next_obs, 'batch'):
                next_obs.batch = None
            
            next_obs = next_obs.to(device)
            
            # 記錄 episode 數據
            episode_rewards.append(reward)
            episode_actions.append(actual_action)
            episode_material_savings.append(env.saved_material_record[-1] if env.saved_material_record else 0)
            
            structure = next_structure
            obs = next_obs
            step_count += 1
        
        # 計算 episode 統計
        total_reward = sum(episode_rewards)
        total_material_saved = sum(episode_material_savings)
        
        # 獲取最終設計的性能指標
        final_performance = {
            'max_stress_ratio': env.static_response_record[-1][0] if env.static_response_record else 0,
            'max_drift_ratio': env.static_response_record[-1][2] if env.static_response_record else 0,
            'min_SCWB_ratio': env.static_response_record[-1][3] if env.static_response_record else 0,
            'material_usage': structure.calculate_material_usage() if hasattr(structure, 'calculate_material_usage') else 0
        }
        
        # 記錄 episode 結果
        episode_result = {
            'episode': episode,
            'total_reward': total_reward,
            'episode_length': step_count,
            'total_material_saved': total_material_saved,
            'final_design': structure.story_level_sections.copy() if hasattr(structure, 'story_level_sections') else [],
            'performance': final_performance,
            'actions_taken': episode_actions
        }
        
        inference_results['episodes'].append(episode_result)
        inference_results['total_rewards'].append(total_reward)
        inference_results['episode_lengths'].append(step_count)
        inference_results['final_designs'].append(structure.story_level_sections.copy() if hasattr(structure, 'story_level_sections') else [])
        inference_results['material_savings'].append(total_material_saved)
        inference_results['performance_metrics'].append(final_performance)
        
        logger.info(f"Inference Episode {episode}: Reward={total_reward:.3f}, Length={step_count}, Material Saved={total_material_saved:.3f}")
    
    return inference_results


def save_inference_results(results: Dict, save_dir: Path, episode_num: int):
    """ 儲存 inference 結果 """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 儲存 JSON 結果
    results_file = save_dir / f"inference_results_episode_{episode_num}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 儲存統計摘要
    summary_file = save_dir / f"inference_summary_episode_{episode_num}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"MuZero Inference Results - Episode {episode_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of episodes: {len(results['episodes'])}\n")
        f.write(f"Average reward: {np.mean(results['total_rewards']):.3f} ± {np.std(results['total_rewards']):.3f}\n")
        f.write(f"Average episode length: {np.mean(results['episode_lengths']):.1f} ± {np.std(results['episode_lengths']):.1f}\n")
        f.write(f"Average material saved: {np.mean(results['material_savings']):.3f} ± {np.std(results['material_savings']):.3f}\n")
        f.write(f"Best reward: {max(results['total_rewards']):.3f}\n")
        f.write(f"Worst reward: {min(results['total_rewards']):.3f}\n")
    
    return results_file, summary_file


def plot_inference_results(results: Dict, save_dir: Path, episode_num: int):
    """ 繪製 inference 結果圖表 """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建圖表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MuZero Inference Results - Episode {episode_num}', fontsize=16)
    
    # 1. 獎勵分佈
    axes[0, 0].hist(results['total_rewards'], bins=10, alpha=0.7, color='blue')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(results['total_rewards']), color='red', linestyle='--', label=f'Mean: {np.mean(results["total_rewards"]):.3f}')
    axes[0, 0].legend()
    
    # 2. Episode 長度分佈
    axes[0, 1].hist(results['episode_lengths'], bins=10, alpha=0.7, color='green')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].set_xlabel('Episode Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(results['episode_lengths']), color='red', linestyle='--', label=f'Mean: {np.mean(results["episode_lengths"]):.1f}')
    axes[0, 1].legend()
    
    # 3. 材料節省分佈
    axes[0, 2].hist(results['material_savings'], bins=10, alpha=0.7, color='orange')
    axes[0, 2].set_title('Material Savings Distribution')
    axes[0, 2].set_xlabel('Material Saved (m³)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(results['material_savings']), color='red', linestyle='--', label=f'Mean: {np.mean(results["material_savings"]):.3f}')
    axes[0, 2].legend()
    
    # 4. 性能指標散點圖
    stress_ratios = [p['max_stress_ratio'] for p in results['performance_metrics']]
    drift_ratios = [p['max_drift_ratio'] for p in results['performance_metrics']]
    axes[1, 0].scatter(stress_ratios, drift_ratios, alpha=0.6)
    axes[1, 0].set_title('Stress vs Drift Ratio')
    axes[1, 0].set_xlabel('Max Stress Ratio')
    axes[1, 0].set_ylabel('Max Drift Ratio')
    
    # 5. 獎勵 vs 材料節省
    axes[1, 1].scatter(results['material_savings'], results['total_rewards'], alpha=0.6)
    axes[1, 1].set_title('Reward vs Material Savings')
    axes[1, 1].set_xlabel('Material Saved (m³)')
    axes[1, 1].set_ylabel('Total Reward')
    
    # 6. SCWB 比率分佈
    scwb_ratios = [p['min_SCWB_ratio'] for p in results['performance_metrics']]
    axes[1, 2].hist(scwb_ratios, bins=10, alpha=0.7, color='purple')
    axes[1, 2].set_title('SCWB Ratio Distribution')
    axes[1, 2].set_xlabel('Min SCWB Ratio')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(np.mean(scwb_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(scwb_ratios):.3f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # 儲存圖表
    plot_file = save_dir / f"inference_plots_episode_{episode_num}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def plot_training_progress(episode_rewards: List[float], episode_lengths: List[int], save_dir: Path, episode_num: int):
    """ 繪製訓練進度圖表 """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'MuZero Training Progress - Episode {episode_num}', fontsize=16)
    
    # 1. Episode Rewards 趨勢
    episodes = list(range(1, len(episode_rewards) + 1))
    axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加移動平均線
    if len(episode_rewards) > 10:
        window_size = min(10, len(episode_rewards) // 10)
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_rewards[start:i+1]))
        axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        axes[0, 0].legend()
    
    # 2. Episode Lengths 趨勢
    axes[0, 1].plot(episodes, episode_lengths, 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Episode Lengths Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加移動平均線
    if len(episode_lengths) > 10:
        window_size = min(10, len(episode_lengths) // 10)
        moving_avg = []
        for i in range(len(episode_lengths)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_lengths[start:i+1]))
        axes[0, 1].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        axes[0, 1].legend()
    
    # 3. Rewards 分佈直方圖
    axes[1, 0].hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Total Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(episode_rewards):.3f}')
    axes[1, 0].axvline(np.median(episode_rewards), color='orange', linestyle='--', 
                       label=f'Median: {np.median(episode_rewards):.3f}')
    axes[1, 0].legend()
    
    # 4. Episode Lengths 分佈直方圖
    axes[1, 1].hist(episode_lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Episode Length Distribution')
    axes[1, 1].set_xlabel('Episode Length')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[1, 1].axvline(np.median(episode_lengths), color='orange', linestyle='--', 
                       label=f'Median: {np.median(episode_lengths):.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 儲存圖表
    plot_file = save_dir / f"training_progress_episode_{episode_num}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def save_training_summary(episode_rewards: List[float], episode_lengths: List[int], save_dir: Path, episode_num: int):
    """ 儲存訓練摘要 """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = save_dir / f"training_summary_episode_{episode_num}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"MuZero Training Summary - Episode {episode_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total episodes: {len(episode_rewards)}\n")
        f.write(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}\n")
        f.write(f"Median reward: {np.median(episode_rewards):.3f}\n")
        f.write(f"Best reward: {max(episode_rewards):.3f}\n")
        f.write(f"Worst reward: {min(episode_rewards):.3f}\n")
        f.write(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}\n")
        f.write(f"Median episode length: {np.median(episode_lengths):.1f}\n")
        f.write(f"Longest episode: {max(episode_lengths)}\n")
        f.write(f"Shortest episode: {min(episode_lengths)}\n")
        
        # 計算最近 10 個 episode 的統計
        if len(episode_rewards) >= 10:
            recent_rewards = episode_rewards[-10:]
            recent_lengths = episode_lengths[-10:]
            f.write(f"\nRecent 10 episodes:\n")
            f.write(f"Average reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}\n")
            f.write(f"Average episode length: {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}\n")
    
    return summary_file


def plot_combined_results(inference_results: Dict, episode_rewards: List[float], episode_lengths: List[int], 
                         save_dir: Path, episode_num: int):
    """ 繪製 inference 和 self-play 的綜合結果 """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建圖表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MuZero Combined Results - Episode {episode_num}', fontsize=16)
    
    # 1. Self-play Episode Rewards 趨勢
    episodes = list(range(1, len(episode_rewards) + 1))
    axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.7, linewidth=1, label='Self-play Rewards')
    axes[0, 0].set_title('Self-play Episode Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加移動平均線
    if len(episode_rewards) > 10:
        window_size = min(10, len(episode_rewards) // 10)
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_rewards[start:i+1]))
        axes[0, 0].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
    axes[0, 0].legend()
    
    # 2. Inference vs Self-play 獎勵比較
    if inference_results['total_rewards']:
        # Self-play 最近的平均獎勵
        recent_self_play_rewards = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
        avg_self_play = np.mean(recent_self_play_rewards)
        std_self_play = np.std(recent_self_play_rewards)
        
        # Inference 的獎勵
        inference_rewards = inference_results['total_rewards']
        avg_inference = np.mean(inference_rewards)
        std_inference = np.std(inference_rewards)
        
        # 繪製比較圖
        categories = ['Self-play (Recent 20)', 'Inference']
        means = [avg_self_play, avg_inference]
        stds = [std_self_play, std_inference]
        
        bars = axes[0, 1].bar(categories, means, yerr=stds, capsize=5, alpha=0.7, 
                              color=['blue', 'orange'])
        axes[0, 1].set_title('Reward Comparison: Self-play vs Inference')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    # 3. Self-play Episode Lengths 趨勢
    axes[0, 2].plot(episodes, episode_lengths, 'g-', alpha=0.7, linewidth=1)
    axes[0, 2].set_title('Self-play Episode Lengths Over Time')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Episode Length')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 添加移動平均線
    if len(episode_lengths) > 10:
        window_size = min(10, len(episode_lengths) // 10)
        moving_avg = []
        for i in range(len(episode_lengths)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_lengths[start:i+1]))
        axes[0, 2].plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        axes[0, 2].legend()
    
    # 4. Inference 獎勵分佈
    if inference_results['total_rewards']:
        axes[1, 0].hist(inference_results['total_rewards'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Inference Reward Distribution')
        axes[1, 0].set_xlabel('Total Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(inference_results['total_rewards']), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(inference_results["total_rewards"]):.3f}')
        axes[1, 0].legend()
    
    # 5. Self-play 獎勵分佈
    axes[1, 1].hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].set_title('Self-play Reward Distribution')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(episode_rewards):.3f}')
    axes[1, 1].axvline(np.median(episode_rewards), color='orange', linestyle='--', 
                       label=f'Median: {np.median(episode_rewards):.3f}')
    axes[1, 1].legend()
    
    # 6. 材料節省比較
    if inference_results['material_savings']:
        recent_self_play_material = [0] * len(recent_self_play_rewards)  # 簡化，實際應該從環境記錄中獲取
        avg_self_play_material = np.mean(recent_self_play_material)
        avg_inference_material = np.mean(inference_results['material_savings'])
        
        categories = ['Self-play (Recent 20)', 'Inference']
        material_means = [avg_self_play_material, avg_inference_material]
        
        bars = axes[1, 2].bar(categories, material_means, alpha=0.7, color=['blue', 'orange'])
        axes[1, 2].set_title('Material Savings Comparison')
        axes[1, 2].set_ylabel('Average Material Saved (m³)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, mean in zip(bars, material_means):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 儲存圖表
    plot_file = save_dir / f"combined_results_episode_{episode_num}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def save_combined_summary(inference_results: Dict, episode_rewards: List[float], episode_lengths: List[int], 
                         save_dir: Path, episode_num: int):
    """ 儲存綜合摘要 """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = save_dir / f"combined_summary_episode_{episode_num}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"MuZero Combined Summary - Episode {episode_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total self-play episodes: {len(episode_rewards)}\n")
        f.write(f"Inference episodes: {len(inference_results['episodes'])}\n")
        
        # Self-play 統計
        f.write(f"\nSelf-play Statistics:\n")
        f.write(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}\n")
        f.write(f"Median reward: {np.median(episode_rewards):.3f}\n")
        f.write(f"Best reward: {max(episode_rewards):.3f}\n")
        f.write(f"Worst reward: {min(episode_rewards):.3f}\n")
        f.write(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}\n")
        
        # 最近 20 個 episode 的統計
        if len(episode_rewards) >= 20:
            recent_rewards = episode_rewards[-20:]
            recent_lengths = episode_lengths[-20:]
            f.write(f"\nRecent 20 self-play episodes:\n")
            f.write(f"Average reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}\n")
            f.write(f"Average episode length: {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}\n")
        
        # Inference 統計
        if inference_results['total_rewards']:
            f.write(f"\nInference Statistics:\n")
            f.write(f"Average reward: {np.mean(inference_results['total_rewards']):.3f} ± {np.std(inference_results['total_rewards']):.3f}\n")
            f.write(f"Average episode length: {np.mean(inference_results['episode_lengths']):.1f} ± {np.std(inference_results['episode_lengths']):.1f}\n")
            f.write(f"Average material saved: {np.mean(inference_results['material_savings']):.3f} ± {np.std(inference_results['material_savings']):.3f}\n")
            f.write(f"Best inference reward: {max(inference_results['total_rewards']):.3f}\n")
            f.write(f"Worst inference reward: {min(inference_results['total_rewards']):.3f}\n")
    
    return summary_file


def main():
    # 解析參數
    args = parse_muzero_args()
    
    # 設定隨機種子
    set_random_seed(args.random_seed)
    
    # 創建檢查點目錄
    args.ckpt_dir = args.ckpt_dir / f'muzero_{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定日誌
    logger = setup_logging(args.ckpt_dir)
    logger.info("Starting MuZero training")
    logger.info(f"Arguments: {args}")
    
    # 設定設備
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # 設定非線性動態分析模擬器
    nda_simulator = None
    nda_norm_dict = None
    DBE_ground_motion_set = None
    MCE_ground_motion_set = None
    if args.do_nonlinear_dynamic_analysis:
        nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
        DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(args.ground_motion_dir, args.ground_motion_number, nda_norm_dict)
    
    # 創建環境
    node_feature_dim = 8 if args.add_structure_geometry else 5
    edge_feature_dim = 13 if args.add_response_features else 11
    
    env_kwargs = {
        "structure_shape": args.structure_shape,
        "add_structure_geometry": args.add_structure_geometry,
        "add_response_features": args.add_response_features,
        "reward_type": args.reward_type,
        "scwb_driven_design": args.scwb_driven_design,
        "do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
        "check_acceleration": args.check_acceleration,
        "check_displacement": args.check_displacement,
        "nda_simulator": nda_simulator,
        "nda_norm_dict": nda_norm_dict,
        "DBE_ground_motion_set": DBE_ground_motion_set,
        "MCE_ground_motion_set": MCE_ground_motion_set,
        "checkpoint_dir": args.ckpt_dir,
        "logger": logger,
        "device": device,
    }
    env = environment.Environment(**env_kwargs)
    
    # 根據不同 structure_shape 計算最大可能的動作數量
    if args.structure_shape == "fixed":
        max_num_actions = 16  # story_num = 4, 4 * 4 = 16
    elif args.structure_shape == "small_random":
        max_num_actions = 16  # story_num = 2-4, 最大 4 * 4 = 16
    elif args.structure_shape == "random":
        max_num_actions = 32  # story_num = 4-7, 最大 7 * 4 = 28，設為 32 保險
    else:
        max_num_actions = 32  # 默認值
    
    # 創建 MuZero 網路
    network = model.MuZeroNetwork(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        max_num_actions=max_num_actions,  # 使用最大動作數量
        num_layers=args.num_layers,
        representation_network_type=args.model_type
    )
    network.to(device)
    
    # 創建 MuZero Agent
    muzero_agent = agent.MuZeroAgent(
        network=network,
        num_simulations=args.muzero_num_simulations,
        discount=args.muzero_discount,
        temperature=args.muzero_temperature,
        device=device
    )
    
    # 創建優化器和緩衝區
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    replay_buffer = buffer.MuZeroReplayBuffer(capacity=args.buffer_capacity)
    
    # 設置環境與 agent 的連接
    env.set_muzero_agent(muzero_agent)
    
    # 訓練迴圈
    episode_rewards = []
    episode_lengths = []
    
    # 在訓練循環中添加定期 inference
    inference_frequency = 50  # 每 50 個 episode 進行一次 inference
    plot_frequency = 20  # 每 20 個 episode 繪製一次訓練進度
    
    # 在訓練循環中添加定期 inference 和統計
    evaluation_frequency = args.evaluation_frequency  # 每 50 個 episode 進行一次評估
    
    for episode in range(args.num_episodes):
        # 階段一：自我對弈產生數據
        episode_length, episode_reward = muzero_self_play(env, muzero_agent, replay_buffer, logger)
        
        # 記錄 episode 結果
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 定期繪製訓練進度
        if (episode + 1) % plot_frequency == 0:
            logger.info(f"Plotting training progress at episode {episode + 1}")
            
            # 繪製訓練進度圖表
            plot_file = plot_training_progress(episode_rewards, episode_lengths, args.ckpt_dir, episode + 1)
            logger.info(f"Training progress plots saved to {plot_file}")
            
            # 儲存訓練摘要
            summary_file = save_training_summary(episode_rewards, episode_lengths, args.ckpt_dir, episode + 1)
            logger.info(f"Training summary saved to {summary_file}")
            
            # 輸出當前統計
            avg_reward = np.mean(episode_rewards[-plot_frequency:]) if len(episode_rewards) >= plot_frequency else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-plot_frequency:]) if len(episode_lengths) >= plot_frequency else np.mean(episode_lengths)
            logger.info(f"Recent {plot_frequency} episodes - Avg Reward: {avg_reward:.3f}, Avg Length: {avg_length:.1f}")
        
        # 階段二：訓練網路
        if episode % args.training_frequency == 0 and len(replay_buffer) >= args.batch_size:
            loss = muzero_train_step(
                network=network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                unroll_steps=args.muzero_unroll_steps,
                discount=args.muzero_discount,
                device=device,
                logger=logger
            )
            
            # 調整溫度參數（隨訓練進度降低）
            new_temperature = max(0.1, args.muzero_temperature * (0.99 ** episode))
            muzero_agent.set_temperature(new_temperature)
            
            logger.info(f"Episode {episode}: Loss={loss:.4f}, Temperature={new_temperature:.3f}")
        
        # 定期進行 inference 評估和統計
        if (episode + 1) % evaluation_frequency == 0:
            logger.info(f"Starting evaluation at episode {episode + 1}")
            
            # 執行 inference
            inference_results = muzero_inference(env, muzero_agent, num_episodes=5, logger=logger)
            
            # 繪製綜合結果圖表
            plot_file = plot_combined_results(inference_results, episode_rewards, episode_lengths, args.ckpt_dir, episode + 1)
            logger.info(f"Combined results plots saved to {plot_file}")
            
            # 儲存綜合摘要
            summary_file = save_combined_summary(inference_results, episode_rewards, episode_lengths, args.ckpt_dir, episode + 1)
            logger.info(f"Combined summary saved to {summary_file}")
            
            # 儲存詳細的 inference 結果
            results_file, inference_summary_file = save_inference_results(inference_results, args.ckpt_dir, episode + 1)
            logger.info(f"Inference results saved to {results_file}")
            
            # 輸出統計摘要
            recent_self_play_rewards = episode_rewards[-evaluation_frequency:] if len(episode_rewards) >= evaluation_frequency else episode_rewards
            avg_self_play = np.mean(recent_self_play_rewards)
            avg_inference = np.mean(inference_results['total_rewards'])
            logger.info(f"Recent {len(recent_self_play_rewards)} self-play episodes - Avg Reward: {avg_self_play:.3f}")
            logger.info(f"Inference episodes - Avg Reward: {avg_inference:.3f}")
            logger.info(f"Performance gap (Inference - Self-play): {avg_inference - avg_self_play:.3f}")
        
        if episode % args.save_frequency == 0:
            # 保存模型
            save_path = args.ckpt_dir / f"muzero_model_episode_{episode}.pt"
            torch.save({
                'episode': episode,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'args': args
            }, save_path)
            logger.info(f"Model saved to {save_path}")
    
    # 訓練結束後繪製最終的綜合結果圖表
    logger.info("Training completed. Generating final combined results plots...")
    final_inference_results = muzero_inference(env, muzero_agent, num_episodes=10, logger=logger)
    final_plot_file = plot_combined_results(final_inference_results, episode_rewards, episode_lengths, args.ckpt_dir, args.num_episodes)
    final_summary_file = save_combined_summary(final_inference_results, episode_rewards, episode_lengths, args.ckpt_dir, args.num_episodes)
    logger.info(f"Final combined results plots saved to {final_plot_file}")
    logger.info(f"Final combined summary saved to {final_summary_file}")
    
    # 保存最終模型
    final_save_path = args.ckpt_dir / "muzero_model_final.pt"
    torch.save({
        'episode': args.num_episodes,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'args': args
    }, final_save_path)
    logger.info(f"Final model saved to {final_save_path}")
    logger.info("MuZero training completed!")


if __name__ == "__main__":
    main() 