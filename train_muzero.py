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
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from collections import Counter, defaultdict

from RL import agent, buffer, environment, model
from Structure import structure
from Validation import check_design, design_space
from Visualization import plot, visualize
from NonlinearDynamicAnalysisSimulator import load_simulator
from Visualization.plot import (
    plot_MuZero_inference_figure, 
    plot_MuZero_selfplay_figure, 
    plot_MuZero_combined_figure
)


def parse_muzero_args() -> argparse.Namespace:
    """解析 MuZero 專用的命令行參數"""
    parser = argparse.ArgumentParser()
    
    # 基本設定
    parser.add_argument("--ckpt_dir", type=Path, default="./Results/MuZero/")
    parser.add_argument("--random_seed", type=int, default=731)
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu, or auto")
    
    # 模型參數
    parser.add_argument("--model_type", type=str, default="Taiwan", help="Taiwan, Japan")
    parser.add_argument("--hidden_dim", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    
    # MuZero 參數
    parser.add_argument("--muzero_num_simulations", type=int, default=30)
    parser.add_argument("--muzero_unroll_steps", type=int, default=3)
    parser.add_argument("--muzero_temperature", type=float, default=1.0)
    parser.add_argument("--muzero_temperature_decay", type=float, default=0.92)
    parser.add_argument("--muzero_discount", type=float, default=0.99)
    
    # 訓練參數
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_capacity", type=int, default=1000)
    parser.add_argument("--training_frequency", type=int, default=5)
    parser.add_argument("--evaluation_frequency", type=int, default=5)
    parser.add_argument("--inference_num", type=int, default=1)
    parser.add_argument("--save_frequency", type=int, default=5)
    
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
    logger.setLevel(logging.DEBUG)
    
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
    """ 執行一步 MuZero 訓練 - 完整 unroll 版本 """

    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0, 0.0  # value, policy, reward, total losses

    network.train()

    games = replay_buffer.sample(batch_size)
    total_value_loss = torch.tensor(0.0, device=device)
    total_policy_loss = torch.tensor(0.0, device=device)
    total_reward_loss = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)

    for game in games:
        try:
            # -------- 1. 決定起點 --------
            if len(game) <= unroll_steps + 1:
                start_index = 0
            else:
                start_index = torch.randint(0, len(game) - unroll_steps - 1, (1,)).item()

            # -------- 2. 初始 observation & hidden --------
            obs = game.observations[start_index].to(device)
            hidden = network.represent(obs)

            # -------- 3. 多步 unroll --------
            value_losses  = []
            policy_losses = []
            reward_losses = []

            bootstrap_discount = 1.0
            for k in range(unroll_steps + 1):
                # 3.1 取當前步驟的合法動作數
                current_num_actions = len(game.policies[start_index + k])

                # 3.2 預測 policy & value
                policy_logits, value_pred = network.predict(hidden, current_num_actions)

                # ----- value loss (使用 N-step return 以反映 bootstrapping) -----
                target_value = calculate_n_step_return(game, start_index + k, unroll_steps, discount)
                # 將目標value也轉換到support space進行比較
                target_value_tensor = torch.tensor([target_value], dtype=torch.float32, device=device)
                target_value_transformed = model.scalar_to_support(target_value_tensor)
                value_losses.append(torch.nn.functional.mse_loss(value_pred.squeeze(), target_value_transformed.squeeze()))

                # ----- policy loss (KL divergence) -----
                target_policy = game.policies[start_index + k].to(device)
                pred_policy   = torch.softmax(policy_logits.squeeze(), dim=-1)
                policy_losses.append(torch.nn.functional.kl_div(torch.log(pred_policy + 1e-8), target_policy + 1e-8, reduction='batchmean'))

                # ----- reward loss (k==0 沒有 reward_pred，因為 represent 不預測 reward) -----
                if k < unroll_steps:
                    action_k = torch.tensor([game.actions[start_index + k]], dtype=torch.long, device=device)
                    next_num_actions = len(game.policies[start_index + k + 1]) if (start_index + k + 1) < len(game.policies) else current_num_actions
                    reward_pred, hidden_next = network.dynamics(hidden, action_k, next_num_actions)

                    # 真實 reward
                    true_reward = torch.tensor([game.rewards[start_index + k]], dtype=torch.float32, device=device)
                    # 將目標reward也轉換到support space進行比較
                    true_reward_transformed = model.scalar_to_support(true_reward)
                    reward_losses.append(torch.nn.functional.mse_loss(reward_pred.squeeze(), true_reward_transformed.squeeze()))

                    # 前進 hidden
                    hidden = hidden_next.detach()  # 避免時間步之間梯度重複計算
                # k == unroll_steps 不需 dynamics

            # -------- 4. 匯總損失 --------
            value_loss  = sum(value_losses)
            policy_loss = sum(policy_losses)
            reward_loss = sum(reward_losses) if reward_losses else torch.tensor(0.0, device=device)
            loss = value_loss + policy_loss + reward_loss

            total_value_loss  += value_loss
            total_policy_loss += policy_loss
            total_reward_loss += reward_loss
            total_loss += loss

            logger.debug(f"Game losses - Value: {value_loss.item():.4f}, Policy: {policy_loss.item():.4f}, " + 
                        f"Reward: {reward_loss.item():.4f}, Total: {loss.item():.4f}")

        except Exception as e:
            logger.warning(f"Error in MuZero train step (unroll): {e}")
            continue

    if total_loss.item() > 0:
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        # 回傳平均 loss
        return (total_value_loss.item() / len(games),
                total_policy_loss.item() / len(games),
                total_reward_loss.item() / len(games),
                total_loss.item() / len(games))
    else:
        return 0.0, 0.0, 0.0, 0.0


def muzero_self_play(env, muzero_agent: agent.MuZeroAgent, replay_buffer: buffer.MuZeroReplayBuffer, logger):
    """ 執行一場 MuZero 自我對弈 """
    # muzero_agent.network.eval()

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

    actions = []
    total_reward = 0
    fail_reason = None
    fail_name = None
    
    logger.info("Starting self-play episode")
    
    while not done and step_count < 1000:  # 限制最大步數避免無限循環
        
            # 獲取合法動作
            legal_actions = env.get_legal_actions(structure)
            
            if not legal_actions:
                logger.info("No legal actions available, episode finished")
                break
            
            # 選擇動作（只考慮合法動作）
            structure_temp = deepcopy(structure)
            actual_action, policy = muzero_agent.select_action(obs, structure, training=True)
            
            # # 將動作索引映射到實際的動作
            # actual_action = legal_actions[action]
            
            # 計算當前狀態的價值預測
            with torch.no_grad():
                hidden_state = muzero_agent.network.represent(obs)
                _, value_pred = muzero_agent.network.predict(hidden_state)
                # 將transformed value轉換回原始scale
                original_value = model.support_to_scalar(value_pred).item()
            
            # 執行動作
            next_structure, reward, done, fail_name, fail_reason = env.step(structure_temp, actual_action)
            next_obs = next_structure.graph.clone()  # 提取下一個狀態的 GraphData
            
            actions.append(actual_action)
            total_reward += reward

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
            current_game.add_step(obs.to('cpu'), actual_action, reward, policy.to('cpu'), done, original_value)
            
            structure = next_structure
            obs = next_obs
            step_count += 1
            
        # except Exception as e:
        #     logger.warning(f"Error in self-play step {step_count}: {e}")
        #     break
        
            # 獲取最終設計的性能指標
            final_performance = {
                'max_stress_ratio': env.static_response_record[-1][0] if env.static_response_record else 0,
                'max_drift_ratio': env.static_response_record[-1][2] if env.static_response_record else 0,
                'min_SCWB_ratio': env.static_response_record[-1][3] if env.static_response_record else 0,
                'material_usage': structure.calculate_material_usage() if hasattr(structure, 'calculate_material_usage') else 0
            }
            
            # 記錄 episode 結果
            episode_result = {
                'total_reward': total_reward,
                'episode_length': step_count,
                'final_design': structure.story_level_sections.copy() if hasattr(structure, 'story_level_sections') else [],
                'performance': final_performance,
                'actions_taken': actions,
                'fail_name': fail_name,
                'fail_reason': fail_reason
            }

    
    # 將遊戲加入緩衝區
    if len(current_game) > 0:
        replay_buffer.push(current_game)
        logger.info(f"Self-play episode finished with {len(current_game)} steps, total reward: {sum(current_game.rewards):.3f}")
    
    return step_count, total_reward, actions, fail_name, fail_reason, episode_result

def muzero_inference(env, muzero_agent: agent.MuZeroAgent, num_episodes: int = 5, logger=None):
    """ 執行 MuZero inference 評估 """
    inference_results = {
        'episodes': [],
        'total_rewards': [],
        'episode_lengths': [],
        'final_designs': [],
        'performance_metrics': []
    }

    # muzero_agent.network.eval()
    logger.info(f"Starting MuZero inference evaluation with {num_episodes} episodes")
    
    for episode in range(num_episodes):
        structure = env.reset(testing=True)
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
        last_fail_name = None
        last_fail_reason = None
        
        while not done and step_count < 1000:
            # 獲取合法動作
            legal_actions = env.get_legal_actions(structure)
            
            if not legal_actions:
                logger.info(f"Episode {episode}: No legal actions available")
                break
            
            # 使用貪婪策略進行 inference
            structure_temp = deepcopy(structure)
            actual_action, _ = muzero_agent.select_action(obs, structure_temp, training=False)
            # actual_action = legal_actions[action]

            if actual_action not in legal_actions:
                logger.error(f"Episode {episode}: Actual action {actual_action} not in legal actions {legal_actions}")
                raise ValueError(f"Actual action {actual_action} not in legal actions {legal_actions}")
            
            # 執行動作
            next_structure, reward, done, fail_name, fail_reason = env.step(structure_temp, actual_action)
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
            last_fail_name = fail_name
            last_fail_reason = fail_reason
            
            structure = next_structure
            obs = next_obs
            step_count += 1
        
        # 計算 episode 統計
        total_reward = sum(episode_rewards)
        
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
            'final_design': structure.story_level_sections.copy() if hasattr(structure, 'story_level_sections') else [],
            'performance': final_performance,
            'actions_taken': episode_actions,
            'fail_name': last_fail_name,
            'fail_reason': last_fail_reason
        }
        
        inference_results['episodes'].append(episode_result)
        inference_results['total_rewards'].append(total_reward)
        inference_results['episode_lengths'].append(step_count)
        inference_results['final_designs'].append(structure.story_level_sections.copy() if hasattr(structure, 'story_level_sections') else [])
        inference_results['performance_metrics'].append(final_performance)
        
        logger.info(f"Inference Episode {episode}: Reward={total_reward:.3f}, Length={step_count}")
    
    return inference_results


def main():
     # === 0. 解析 CLI & init ========================================================================
    args = parse_muzero_args()
    set_random_seed(args.random_seed)

    # 初始化檔案夾與 Logger
    ts = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    ckpt_dir = args.ckpt_dir / f"muzero_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    # 子資料夾：models / plots / action_seq
    models_dir = ckpt_dir / 'models'
    plots_dir = ckpt_dir / 'plots'
    for d in [models_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # plots 下進一步拆分
    selfplay_plots = plots_dir / 'selfplay'
    inference_plots = plots_dir / 'inference'
    combined_plots  = plots_dir / 'combined'
    for d in [selfplay_plots, inference_plots, combined_plots]:
        d.mkdir(exist_ok=True)

    logger = setup_logging(ckpt_dir)
    logger.info("Starting MuZero training")
    logger.info(args)
    
    # === 1. 建立環境 & agent ========================================================================
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
        "restrict_action": args.restrict_action,
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
        story_num = 4
    elif args.structure_shape == "small_random":
        max_num_actions = 16  # story_num = 2-4, 最大 4 * 4 = 16
        story_num = 4
    elif args.structure_shape == "random":
        max_num_actions = 32  # story_num = 4-7, 最大 7 * 4 = 28，設為 32 保險
        story_num = 7
    else:
        max_num_actions = 32  # 默認值
        story_num = 7
    
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
        env=env,
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

    # === 2. 訓練 loop ==============================================================================
    # 歷史記錄
    sp_rewards, sp_lengths, sp_actions, sp_fails = [], [], [], []
    inf_rewards, inf_lengths, inf_actions, inf_fails = [], [], [], []
    selfplay_record = []
    inference_record = []
    loss_record = []

    best_inf_score = -float('inf')
    best_inf_ep    = None

    for ep in range(args.num_episodes):
        # --- Self-play ---
        length, reward, actions_seq, fail_name, fail_reason, rec = muzero_self_play(env, muzero_agent, replay_buffer, logger)
        sp_lengths.append(length)
        sp_rewards.append(reward)
        sp_actions.append(actions_seq)
        sp_fails.append(fail_reason)
        selfplay_record.append(rec)

        # 2) Train step
        if ep % args.training_frequency == 0 and len(replay_buffer) >= args.batch_size:
            value_loss, policy_loss, reward_loss, total_loss = muzero_train_step(network, optimizer, replay_buffer,
                                     args.batch_size, args.muzero_unroll_steps,
                                     args.muzero_discount, device, logger)
            logger.info(f"Episode {ep}: train loss={total_loss:.4f}, value_loss={value_loss:.4f}, policy_loss={policy_loss:.4f}, reward_loss={reward_loss:.4f}")
            loss_record.append({'episode': ep, 'value_loss': value_loss, 'policy_loss': policy_loss, 'reward_loss': reward_loss, 'total_loss': total_loss})
            
            # 使用 plot.py 中的函數繪製 loss history
            from Visualization.plot import plot_muzero_losses
            plot_muzero_losses(loss_record, plots_dir)

        # 3) Evaluation & Plotting
        if ep % args.evaluation_frequency == 0:
            rnd = ep // args.evaluation_frequency
            # 執行 Inference (同樣收集 actions & fail_reasons)
            inf_res = muzero_inference(env, muzero_agent, num_episodes=args.inference_num, logger=logger)
            # 假設 muzero_inference 返回 dict 含 'episodes' list，並可提取 actions & fail_reasons
            batch_rewards = inf_res['total_rewards']
            batch_lengths = inf_res['episode_lengths']
            batch_actions = [ep['actions_taken'] for ep in inf_res['episodes']]
            batch_fails   = [ep['fail_reason'] for ep in inf_res['episodes']]


            inf_rewards.extend(batch_rewards)
            inf_lengths.extend(batch_lengths)
            inf_actions.extend(batch_actions)
            inf_fails.extend(batch_fails)
            inference_record.append(inf_res)

            avg_inf_score = np.mean(batch_rewards)
            if avg_inf_score > best_inf_score:
                best_inf_score = avg_inf_score
                best_inf_ep    = ep
                best_path = models_dir / f"best_inference_ep{ep}.pt"
                torch.save({
                    'episode': ep,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_inf_score': best_inf_score
                }, best_path)
                logger.info(f"New best inference: score={best_inf_score:.3f} at (global ep={ep}); model saved to {best_path}")

            # 繪圖
            # 紀錄 self-play action sequence 圖
            plot_MuZero_selfplay_figure(
                out_path = ckpt_dir / 'plots' / 'selfplay' / f"selfplay_ep_{ep}_rnd_{rnd}_{ts}.png",
                reward_history = sp_rewards,
                length_history = sp_lengths,
                best_actions_history = sp_actions,
                rewards_this_round = sp_rewards[-args.evaluation_frequency:],
                fail_reasons_this_round = sp_fails[-args.evaluation_frequency:],
                story_num = story_num
            )
            # 繪製 inference
            plot_MuZero_inference_figure(
                out_path = ckpt_dir / 'plots' / 'inference' / f"inference_ep_{ep}_rnd_{rnd}_{ts}.png",
                reward_history = inf_rewards,
                length_history = inf_lengths,
                best_actions_history = inf_actions,
                rewards_this_round = batch_rewards,
                fail_reasons_this_round = batch_fails,
                story_num = story_num
            )

            # 繪製 combined
            plot_MuZero_combined_figure(
                out_path = ckpt_dir / 'plots' / 'combined' / f"combined_ep_{ep}_rnd_{rnd}_{ts}.png",
                selfplay_reward_hist = sp_rewards,
                selfplay_length_hist = sp_lengths,
                inference_reward_hist = inf_rewards,
                inference_length_hist = inf_lengths,
                frequency = args.evaluation_frequency
            )

        # 4) Save checkpoint
        if ep % args.save_frequency == 0:
            ckpt = models_dir / f"model_ep_{ep}.pt"
            torch.save({'ep':ep, 'net':network.state_dict(), 'opt':optimizer.state_dict()}, ckpt)
            logger.info(f"Saved checkpoint to {ckpt}")

            # 記錄 selfplay 和 inference 的結果
            rec = {
                'selfplay': selfplay_record,
                'inference': inference_record
            }
            with open(ckpt_dir / 'record.json', 'w') as f:
                json.dump(rec, f)
        
        # 溫度 decay，每 10 個 episode 調整一次
        if (ep + 1) % 10 == 0:
            new_temp = max(0.1, muzero_agent.temperature * args.muzero_temperature_decay)
            muzero_agent.set_temperature(new_temp)
            logger.info(f"Decay temperature to {new_temp:.3f}")

    logger.info("Training finished.")

if __name__ == "__main__":
    main()
