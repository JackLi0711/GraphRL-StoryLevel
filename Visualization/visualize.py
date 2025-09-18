import os
import glob
import torch
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

from PIL import Image
from pathlib import Path
from logging import Logger
from copy import deepcopy
from torch_geometric.loader import DataLoader
from collections import defaultdict
import json

from RL import agent, environment
from Structure import structure, pisa


def make_json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable ones."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj
from Structure.sections import beam_sections, column_sections


def action_to_story_component(action_index: int, structure: structure.Structure) -> tuple:
    """
    將 action index 映射到 (樓層號, 構件類型)。

    Args:
        action_index: Action 索引
        structure: 結構對象

    Returns:
        tuple: (樓層號, 構件類型)
    """
    category = structure.story_level_categories[action_index]

    if category == 'xdir_beam':
        story_index = action_index
        component_type = 'X-Beam'
    elif category == 'zdir_beam':
        story_index = action_index - len(structure.story_xdir_beam_member)
        component_type = 'Z-Beam'
    elif category == 'outer_column':
        story_index = action_index - len(structure.story_xdir_beam_member) - len(structure.story_zdir_beam_member)
        component_type = 'Outer-Column'
    else:  # inner_column
        story_index = action_index - len(structure.story_xdir_beam_member) - len(structure.story_zdir_beam_member) - len(structure.story_outer_column_member)
        component_type = 'Inner-Column'

    return story_index + 1, component_type  # 樓層從1開始


def analyze_actions_by_story_component(action_indices: list, structure: structure.Structure) -> dict:
    """
    分析 actions 按樓層和構件類型的統計。

    Args:
        action_indices: Action 索引列表
        structure: 結構對象

    Returns:
        dict: 統計結果
    """
    stats = {
        'total_actions': len(action_indices),
        'by_story': {},
        'by_component': {'Inner-Column': 0, 'Outer-Column': 0, 'X-Beam': 0, 'Z-Beam': 0}
    }

    for action_idx in action_indices:
        story, component = action_to_story_component(action_idx, structure)

        if story not in stats['by_story']:
            stats['by_story'][story] = {}
        if component not in stats['by_story'][story]:
            stats['by_story'][story][component] = 0

        stats['by_story'][story][component] += 1
        stats['by_component'][component] += 1

    return stats


def prepare_episode_option_data(episode_data: dict, structure: structure.Structure) -> list:
    """
    準備 episode 中每個 option instance 的詳細數據。

    Args:
        episode_data: Episode 數據包含 actions, options, option_instances
        structure: 結構對象

    Returns:
        list: 按時間順序排列的 (instance_id, instance_data) 對
    """
    actions = episode_data['actions']
    options = episode_data['options']
    option_instances = episode_data['option_instances']

    # 按 instance_id 分組
    instance_groups = {}
    for instance_data in option_instances:
        instance_id = instance_data['instance_id']
        if instance_id not in instance_groups:
            instance_groups[instance_id] = {
                'option_index': instance_data['option_index'],
                'action_indices': [],
                'first_action_step': float('inf')
            }

        action_step = instance_data['action_step']
        instance_groups[instance_id]['action_indices'].append(actions[action_step])
        instance_groups[instance_id]['first_action_step'] = min(
            instance_groups[instance_id]['first_action_step'], action_step
        )

    # 為每個 instance 生成統計
    for instance_id in instance_groups:
        group_actions = instance_groups[instance_id]['action_indices']
        stats = analyze_actions_by_story_component(group_actions, structure)
        instance_groups[instance_id]['statistics'] = stats

    # 按首次出現順序排序
    sorted_instances = sorted(instance_groups.items(),
                            key=lambda x: x[1]['first_action_step'])

    return sorted_instances


def _add_option_preview_panel(fig, preview_data: dict):
    """
    在右側添加 option actions 預告統計面板。

    Args:
        fig: matplotlib figure 對象
        preview_data: 預告數據包含 option_index, instance_id, statistics
    """
    ax_preview = fig.add_subplot(1, 3, 3)
    ax_preview.set_axis_off()

    option_idx = preview_data['option_index']
    instance_id = preview_data['instance_id']
    stats = preview_data['statistics']

    # Title
    title_text = f"Option {option_idx} (Instance {instance_id})\nUpcoming Actions"
    ax_preview.text(0.5, 0.95, title_text, ha='center', va='top',
                   fontsize=16, weight='bold', transform=ax_preview.transAxes)

    # Total information
    total_actions = stats['total_actions']
    summary_text = f"Total: {total_actions} actions\n\n"

    # 按樓層統計
    y_pos = 0.85
    ax_preview.text(0.05, y_pos, summary_text, fontsize=14,
                   transform=ax_preview.transAxes, weight='bold')

    y_pos -= 0.08
    for story in sorted(stats['by_story'].keys()):
        story_text = f"{story}F:"
        ax_preview.text(0.05, y_pos, story_text, fontsize=13, weight='bold',
                       transform=ax_preview.transAxes)
        y_pos -= 0.04

        for component, count in stats['by_story'][story].items():
            component_text = f"  • {component}: {count} times"
            ax_preview.text(0.1, y_pos, component_text, fontsize=16,
                          transform=ax_preview.transAxes)
            y_pos -= 0.04
        y_pos -= 0.02  # 樓層間距

    # 添加邊框
    from matplotlib.patches import Rectangle
    bbox = Rectangle((0.02, 0.02), 0.96, 0.96,
                    transform=ax_preview.transAxes,
                    fill=False, edgecolor='blue', linewidth=2)
    ax_preview.add_patch(bbox)


def _generate_option_preview_gif(save_dir: Path, episode_info: dict) -> Path:
    """
    生成 option preview GIF 動畫。

    Args:
        save_dir: 保存目錄
        episode_info: Episode 信息

    Returns:
        Path: GIF 文件路徑
    """
    # 收集所有 PNG 文件，按數字順序排序
    frame_files = []
    for png_file in save_dir.glob("*.png"):
        try:
            frame_num = int(png_file.stem)
            frame_files.append((frame_num, png_file))
        except ValueError:
            # 跳過非數字命名的文件
            continue

    # 按幀號排序
    frame_files.sort(key=lambda x: x[0])
    frame_paths = [f[1] for f in frame_files]

    if not frame_paths:
        print(f"No frames found in {save_dir}")
        return None

    frames = [Image.open(frame_path) for frame_path in frame_paths]

    # 延長最後一幀的顯示時間
    if frames:
        last_frame = frames[-1]
        frames.extend([last_frame] * 3)

        gif_name = f"option_preview_round{episode_info['round_number']}_score{episode_info['score']:.2f}.gif"
        gif_path = save_dir / gif_name

        frames[0].save(
            gif_path, format="GIF", append_images=frames[1:],
            save_all=True, duration=1500, loop=0
        )

        print(f"Generated GIF: {gif_path}")
        return gif_path

    return None


def visualize_option_preview_process(agent: agent.DeepQAgent,
                                   env: environment.Environment,
                                   logger: Logger,
                                   save_model_path: Path,
                                   episode_data: dict,
                                   episode_info: dict,
                                   save_base_dir: Path):
    """
    以 option 為單位進行可視化，每個 option instance 開始前顯示結構狀態和即將執行的 actions 統計。

    Args:
        agent: DeepQAgent 對象
        env: Environment 對象
        logger: Logger 對象
        save_model_path: 模型路徑
        episode_data: Episode 數據包含 actions, options, option_instances
        episode_info: Episode 信息包含 episode_number, round_number, score
        save_base_dir: 基礎保存目錄
    """
    logger.info(f"Generating option preview visualization for round {episode_info['round_number']}")

    # 創建帶 episode 信息的目錄
    save_dir_name = f"option_preview_process_round{episode_info['round_number']}_ep{episode_info['episode_number']}_score{episode_info['score']:.2f}"
    save_dir = save_base_dir / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加載最佳驗證模型
    if save_model_path.exists():
        checkpoint = torch.load(save_model_path, map_location=torch.device(agent.device))
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

        # 處理不同的 checkpoint 格式
        if 'state_dict' in checkpoint:
            # 新格式：使用 'state_dict' 保存整個模型狀態
            if hasattr(agent, 'gnn'):
                agent.gnn.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded Option-Critic model state from checkpoint (new format)")
        elif 'Q.weight' in checkpoint:
            # 舊格式：直接保存模型參數（沒有包裝在 state_dict 中）
            if hasattr(agent, 'gnn'):
                agent.gnn.load_state_dict(checkpoint)
                logger.info("Loaded Option-Critic model state from checkpoint (old format)")
        elif 'gnn' in checkpoint:
            # 分別保存的組件格式
            if hasattr(agent, 'gnn'):
                agent.gnn.load_state_dict(checkpoint['gnn'])
            if hasattr(agent, 'online_q_network'):
                agent.online_q_network.load_state_dict(checkpoint['online_q_network'])
            if hasattr(agent, 'target_q_network'):
                agent.target_q_network.load_state_dict(checkpoint['target_q_network'])
            logger.info("Loaded model components from checkpoint")
        else:
            logger.warning(f"Unexpected checkpoint format. Available keys: {list(checkpoint.keys())}")
            logger.info("Using current model weights without loading checkpoint")
    else:
        logger.warning(f"Model file {save_model_path} not found, using current model weights")

    # 重置環境到初始狀態
    structure = env.reset(testing=True)

    # 準備 option instance 數據
    sorted_option_instances = prepare_episode_option_data(episode_data, structure)

    actions = episode_data['actions']
    total_actions = len(actions)
    action_step = 0

    logger.info(f"Processing {len(sorted_option_instances)} option instances with {total_actions} total actions")

    # 為每個 option instance 生成預告圖
    for option_frame_idx, (instance_id, instance_data) in enumerate(sorted_option_instances):
        logger.info(f"Generating preview for option instance {instance_id} (frame {option_frame_idx})")

        # 準備預告統計數據
        preview_data = {
            'option_index': instance_data['option_index'],
            'instance_id': instance_id,
            'statistics': instance_data['statistics']
        }

        # 生成當前結構的 Q-values (針對 Option-Critic 模型)
        with torch.no_grad():
            graph = structure.graph.clone()
            device = agent.device
            graph = graph.to(device)

            # 對於 Option-Critic，我們需要處理 option Q-values
            if hasattr(agent, 'gnn') and agent.gnn is not None:
                # 使用 Option-Critic 的方式獲取狀態
                story_level_state, global_state = agent.gnn.get_state(
                    graph.x, graph.edge_index, graph.edge_attr,
                    structure.aux["story_batch"].to(device), None
                )
                # 獲取 option Q-values
                option_q_values = agent.gnn.get_Q(global_state)
                # 為了與原可視化兼容，我們創建一個偽造的 story-level Q-values
                q_values = torch.zeros(len(structure.story_level_actions), device=device)
                # 可以選擇將 option Q-values 的平均值分配到所有 story actions
                if option_q_values.numel() > 0:
                    avg_q = option_q_values.mean().item()
                    q_values.fill_(avg_q)
            else:
                q_values = torch.zeros(len(structure.story_level_actions), device=device)

        # 生成預告圖片
        vis_path = save_dir / f"{option_frame_idx}.png"
        accumulated_reward = sum([env.material_usage_record[i] - env.material_usage_record[i+1]
                                for i in range(min(len(env.material_usage_record)-1, action_step))])

        _visualize_one_iteration(
            structure=structure,
            iteration=option_frame_idx,
            env=env,
            accumulated_reward=accumulated_reward,
            q_values=q_values,
            save_fig_path=vis_path,
            option_action_preview=preview_data
        )

        # 執行該 option instance 的所有 actions，更新結構狀態以準備下一個 option
        for action_idx in instance_data['action_indices']:
            if action_step < total_actions:
                try:
                    structure, reward, done, fail_name, fail_reason = env.step(structure, action_idx)
                    action_step += 1
                    logger.debug(f"Applied action {action_idx} (step {action_step}/{total_actions})")
                except Exception as e:
                    logger.error(f"Error applying action {action_idx}: {e}")
                    break

    # 生成 episode 摘要 JSON
    summary_data = {
        'episode_info': episode_info,
        'total_option_instances': len(sorted_option_instances),
        'total_actions': total_actions,
        'option_instances': [
            {
                'instance_id': instance_id,
                'option_index': instance_data['option_index'],
                'num_actions': len(instance_data['action_indices']),
                'statistics': instance_data['statistics']
            }
            for instance_id, instance_data in sorted_option_instances
        ]
    }

    summary_path = save_dir / "episode_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(summary_data), f, indent=2, ensure_ascii=False)

    # 生成 GIF 動畫
    gif_path = _generate_option_preview_gif(save_dir, episode_info)
    if gif_path:
        logger.info(f"Generated option preview GIF: {gif_path}")
    else:
        logger.warning("Failed to generate GIF animation")

    logger.info(f"Option preview visualization completed. Results saved to: {save_dir}")


@torch.no_grad()
def visualize_edge_embedding(agent, env, logger, ckpt_dir):
    # Import TSNE only when needed to avoid import errors if sklearn is not available
    from sklearn.manifold import TSNE

    logger.info(f"Visualizing edge embedding tsne......")
    # Make directory
    save_dir = ckpt_dir / "edge_embedding_tsne"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # load best-validation model
    save_model_path = ckpt_dir / "model.pt"
    checkpoint = torch.load(save_model_path, map_location=torch.device(agent.device))
    agent.gnn.load_state_dict(checkpoint['gnn'])    
    agent.online_q_network.load_state_dict(checkpoint['online_q_network'])    
    agent.target_q_network.load_state_dict(checkpoint['target_q_network']) 

    # get testing structure & graph
    device = agent.device
    structure = env.reset(testing=True)
    graph = structure.graph.clone()

    # collect the embeddings from edges every 5 timesteps
    edge_embeddings = torch.tensor([]).to(device)
    features = {}
    feature_list = ["is_column", "is_beam", "length", "A", "Iz", "Iy", "Zz", "if_fixed", "if_top", "if_side", "beta_x", "beta_z"]
    if env.add_structure_geometry:
        feature_list += ["y_grid_num", "y_coord", "y_ratio"]
    for feature_name in feature_list:
        features[feature_name] = torch.tensor([]).to(device)
    done = None
    score = 0
    timestep = 0
    while not done:
        # go through gnn and get embedding before q-network
        graph = graph.to(device)
        state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(device), None)

        if timestep % 1 == 0:
            edge_embed_dim = state.shape[1] // 2
            edge_embed = state[:, :edge_embed_dim]
            edge_embeddings = torch.cat([edge_embeddings, edge_embed], dim=0)

            # record the edge/node features
            edge_feature = graph.edge_attr[::2, :]
            features["is_column"] = torch.cat([features["is_column"], edge_feature[:, 0]], dim=0)
            features["is_beam"] = torch.cat([features["is_beam"], edge_feature[:, 1]], dim=0)
            features["length"] = torch.cat([features["length"], edge_feature[:, 2]], dim=0)
            features["A"] = torch.cat([features["A"], (edge_feature[:, 3] + edge_feature[:, 7]) / 2], dim=0)
            features["Iz"] = torch.cat([features["Iz"], (edge_feature[:, 4] + edge_feature[:, 8]) / 2], dim=0)
            features["Iy"] = torch.cat([features["Iy"], (edge_feature[:, 5] + edge_feature[:, 9]) / 2], dim=0)
            features["Zz"] = torch.cat([features["Zz"], (edge_feature[:, 6] + edge_feature[:, 10]) / 2], dim=0)
            row, col = graph.edge_index
            node2edge = (graph.x[row] + graph.x[col])[::2, :] / 2
            features["if_fixed"] = torch.cat([features["if_fixed"], node2edge[:, 0]], dim=0)
            features["if_top"] = torch.cat([features["if_top"], node2edge[:, 1]], dim=0)
            features["if_side"] = torch.cat([features["if_side"], node2edge[:, 2]], dim=0)
            features["beta_x"] = torch.cat([features["beta_x"], node2edge[:, 3]], dim=0)
            features["beta_z"] = torch.cat([features["beta_z"], node2edge[:, 4]], dim=0)
            if env.add_structure_geometry:
                features["y_grid_num"] = torch.cat([features["y_grid_num"], node2edge[:, 5]], dim=0)
                features["y_coord"] = torch.cat([features["y_coord"], node2edge[:, 6]], dim=0)
                features["y_ratio"] = torch.cat([features["y_ratio"], node2edge[:, 7]], dim=0)

        # select action and update structure
        dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else None
        action, _ = agent.choose_action(state, 
                                        structure.already_minimum_section_story_indexes,
                                        dont_select_story_member_indexes, 
                                        greedy=True)
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # get next state
        graph = structure.graph.clone()
        timestep += 1
        score += reward
        logger.info(f"timestep: {timestep}, accumulated_score: {score}")


    edge_embeddings = edge_embeddings.cpu().numpy()
    print("edge embedding shape:", edge_embeddings.shape)
    for key in features:
        features[key] = features[key].cpu().numpy()
        print(f"{key} shape:", features[key].shape)

    
    def plot_TSNE(tsne, label, color, name):
        plt.figure(figsize=(10,10), facecolor="w")
        plt.scatter(tsne[:,0], tsne[:,1], c=label, s=100, alpha=0.75, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar()
        cbar.set_label(color, size=15,)
        plt.savefig(save_dir / f"{name}.png")
        plt.close()
        
    tsne = TSNE(n_components=2, init='random', perplexity=10, n_iter=1000).fit_transform(edge_embeddings)

    for key in features:
        plot_TSNE(tsne, features[key], color=key, name=key)




def _visualize_one_iteration(structure: structure.Structure,
                             iteration: int,
                             env: environment.Environment,
                             accumulated_reward: float,
                             q_values: torch.Tensor,
                             save_fig_path: Path,
                             option_action_preview: dict = None):
    # plot 3d - 調整佈局和大小以容納 option preview
    if option_action_preview:
        fig = plt.figure(figsize=(30, 15), facecolor="w")  # 加寬以容納統計面板
        q_subplot_spec = (1, 3, 1)
        structure_subplot_spec = (1, 3, 2)
    else:
        fig = plt.figure(figsize=(20, 15), facecolor="w")  # 原有佈局
        q_subplot_spec = (1, 2, 1)
        structure_subplot_spec = (1, 2, 2)

    # First plot q_values
    ax = fig.add_subplot(*q_subplot_spec, projection="3d", facecolor="w")
    ax.set_axis_off()

    # noamalize q_values
    q_values = q_values.detach().cpu().numpy()
    q_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))

    q_values_member = [0 for _ in range(structure.member_number)]
    for i in range(q_values.shape[0]):
        for member_index in structure.story_level_actions[i]:
            q_values_member[member_index] = q_values[i]

    # colormap
    cmap = matplotlib.cm.get_cmap("Greys")

    # plot nodes
    x_coord, y_coord, z_coord = [], [], []
    for node_index in range(len(structure.node_coord_dict.keys())):
        node_name = f"N{node_index+1}"
        x, y, z = structure.node_coord_dict[node_name]
        x_coord.append(x)
        y_coord.append(y)
        z_coord.append(z)
    ax.scatter(x_coord, z_coord, y_coord, color="black", s=150, alpha=0.75, edgecolor="black")
    ax.set_box_aspect((np.ptp(x_coord), np.ptp(z_coord), np.ptp(y_coord)))

    # plot edges
    edges_coord = []
    edges_color = []
    for member_name in structure.member_to_nodeIndex_dict.keys():
        # member coord
        elems = structure.member_to_nodeIndex_dict[member_name]
        node1_index, node2_index = elems[:2]
        node1_name, node2_name = f"N{node1_index+1}", f"N{node2_index+1}"
        x1, y1, z1 = structure.node_coord_dict[node1_name]
        x2, y2, z2 = structure.node_coord_dict[node2_name]
        edges_coord.append([x1, y1, z1, x2, y2, z2])
        # member q value
        member_index = int(member_name[1:]) - 1
        if member_index in structure.already_minimum_section_story_indexes:
            q_val = 0
        else:
            q_val = q_values_member[member_index]
        color = cmap(q_val)[:3]
        edges_color.append(color)

    for edge_i in range(len(edges_coord)):
        x1, y1, z1, x2, y2, z2 = edges_coord[edge_i]
        xx = [x1, x2]
        yy = [y1, y2]
        zz = [z1, z2]
        color = edges_color[edge_i]
        # plot a border on a line: https://stackoverflow.com/questions/12729529/can-i-give-a-border-outline-to-a-line-in-matplotlib-plot-function
        ax.plot(xx, zz, yy, c=(color), linewidth=5, path_effects=[pe.Stroke(linewidth=9, foreground='black'), pe.Normal()])

    ax.set_title(f"Q values", fontsize=30)


    # Then plot beam-column sections
    ax = fig.add_subplot(*structure_subplot_spec, projection="3d", facecolor="w")
    ax.set_axis_off()

    # plot nodes
    x_coord, y_coord, z_coord = [], [], []
    for node_index in range(len(structure.node_coord_dict.keys())):
        node_name = f"N{node_index+1}"
        x, y, z = structure.node_coord_dict[node_name]
        x_coord.append(x)
        y_coord.append(y)
        z_coord.append(z)
    ax.scatter(x_coord, z_coord, y_coord, color="black", s=150, alpha=0.75, edgecolors="black")
    ax.set_box_aspect((np.ptp(x_coord), np.ptp(z_coord), np.ptp(y_coord)))

    # plot edges
    edges_coord = []
    edges_color = []
    for member_name in structure.member_to_nodeIndex_dict.keys():
        # member coord
        elems = structure.member_to_nodeIndex_dict[member_name]
        node1_index, node2_index = elems[:2]
        node1_name, node2_name = f"N{node1_index+1}", f"N{node2_index+1}"
        x1, y1, z1 = structure.node_coord_dict[node1_name]
        x2, y2, z2 = structure.node_coord_dict[node2_name]
        edges_coord.append([x1, y1, z1, x2, y2, z2])
        # member section
        member_category = structure.member_category_dict[member_name]
        member_section = structure.member_section_dict[member_name]
        section = column_sections[member_section] if member_category == 'y' else beam_sections[member_section]
        color = section["color"]
        color = (color[0]/255, color[1]/255, color[2]/255)
        edges_color.append(color)

    for edge_i in range(len(edges_coord)):
        x1, y1, z1, x2, y2, z2 = edges_coord[edge_i]
        xx = [x1, x2]
        yy = [y1, y2]
        zz = [z1, z2]
        color = edges_color[edge_i]
        ax.plot(xx, zz, yy, c=(color), linewidth=5)

        
    # total_saved_material == total_material_difference
    saved_material = np.sum(env.saved_material_record)
    saved_material_SCWB = np.sum(env.saved_material_record_SCWB)
    #total_material_difference = env.material_usage_record[0] - env.material_usage_record[-1]

    if env.do_nonlinear_dynamic_analysis and "acceleration" in env.reward_type:
        acc_record_x = np.array(env.acc_record['X-dir'])
        acc_record_z = np.array(env.acc_record['Z-dir'])
        total_acc_decrement = np.sum(acc_record_x[0,:] - acc_record_x[-1,:]) + np.sum(acc_record_z[0,:] - acc_record_z[-1,:])
    else: 
        total_acc_decrement = 0

    infos = f"material usage: {env.material_usage_record[-1]:5.2f} m3\n" + f"reduced material: {saved_material:5.2f} m3\n"
    if env.scwb_driven_design:
        infos += f"reduced material(SCWB): {saved_material_SCWB:5.2f} m3"
    title = f"Reward: {env.reward_type}\n" + f"Iteration: {iteration:4d}\n" + infos

    ax.set_title(title, fontsize=30)

    # 如果有 option preview 數據，添加統計面板
    if option_action_preview:
        _add_option_preview_panel(fig, option_action_preview)

    fig.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()


def _frames_to_video(ckpt_dir: Path, 
                     frame_dir: Path, 
                     testing: bool=True, 
                     taller: bool=False, 
                     short: bool=False, 
                     chances: int=0):
    frame_names = [image for image in glob.glob(f"{frame_dir}/*.png")]
    # adjust the order
    frame_names = [str(frame_dir / f"{i}.png") for i in range(0, len(frame_names))]
    if short:
        frame_names = frame_names[::10] + [frame_names[-1]]
    print("frame names length:")
    print(len(frame_names))
    frames = [Image.open(name) for name in frame_names]
    last_frame = frames[-1]
    frames += [last_frame] * 5
    
    # frame_duration_ms = 1000 if not short else 200
    frame_duration_ms = 200
    frame_one = frames[0]
    if testing:
        animation_name = f"design_animation_testing_{chances}chance.gif"
    elif taller:
        animation_name = f"design_animation_taller_{chances}chance.gif"
    elif short:
        animation_name = f"design_animation_short_{chances}chance.gif"
    else:
        animation_name = f"design_animation_random_{chances}chance.gif"
        
    frame_one.save(ckpt_dir / animation_name, format="GIF", append_images=frames, save_all=True, duration=frame_duration_ms, loop=0)


def visualize_design_process(agent: agent.DeepQAgent, 
                             env: environment.Environment, 
                             logger: Logger, 
                             save_model_path: Path, 
                             testing_structure: bool=False, 
                             taller_structure: bool=False, 
                             initial_design: list[int]=None,
                             chances: int=0):
    logger.info(f"Visualizing design process......")
    
    # Make directory
    original_chances = chances
    if testing_structure:
        save_dir = env.checkpoint_dir / f"design_process_testing_{original_chances}chance"
    elif taller_structure:
        save_dir = env.checkpoint_dir / f"design_process_taller_{original_chances}chance"
    else:
        save_dir = env.checkpoint_dir / f"design_peocess_random_{original_chances}chance"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # load best-validation model
    checkpoint = torch.load(save_model_path, map_location=torch.device(agent.device))
    agent.gnn.load_state_dict(checkpoint["gnn"])    
    agent.online_q_network.load_state_dict(checkpoint["online_q_network"])    
    agent.target_q_network.load_state_dict(checkpoint["target_q_network"]) 
    
    # get testing structure & graph
    device = agent.device
    if testing_structure:
        structure = env.reset(testing=True, initial_design=initial_design)
    elif taller_structure:
        structure = env.reset(taller=True, initial_design=initial_design)
    else:
        structure = env.reset(initial_design=initial_design)
    graph = structure.graph.clone()

    done = None
    accumulated_reward = 0
    timestep = 0
    action_list = []
    while not done:
        original_structure = deepcopy(structure)
        # go through gnn and get embedding before q-network
        with torch.no_grad():
            graph = graph.to(device)
            state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(device), None)

        # select action and update structure
        print(f"story level sections: {structure.story_level_sections}")
        print(f"original minimum: {structure.already_minimum_section_story_indexes}")
        print(f"restrict actions: {structure.restrict_action_space()}")
        
        dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else []
        action, _ = agent.choose_action(state, 
                                        structure.already_minimum_section_story_indexes,
                                        dont_select_story_member_indexes, 
                                        greedy=True)

        # visualize
        vis_path = save_dir / f"{timestep}.png"
        q_values = agent.online_q_network(state)
        _visualize_one_iteration(structure, timestep, env, accumulated_reward, q_values, vis_path)

        # update action
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        dont_select_during_cahnce_loop = []
        while done and chances > 0:
            print(f"----- Fails, current failed action: {action:3d}, current chances: {chances:3d}")
            
            # remove the record of current design which didn't pass the constraints
            if env.do_nonlinear_dynamic_analysis and "acceleration" in env.reward_type:
                env.acc_record['X-dir'].pop(-1)
                env.acc_record['Z-dir'].pop(-1)
            
            # if greedy will fail, choose the subgreedy
            structure = deepcopy(original_structure)
            
            dont_select_during_cahnce_loop.append(action)
            print(f"dont select during chance loop: {dont_select_during_cahnce_loop}")
            
            print(f"story level sections: {structure.story_level_sections}")
            print(f"original minimum: {structure.already_minimum_section_story_indexes}")
            print(f"restrict actions: {structure.restrict_action_space()}")

            if agent.restrict_action:
                dont_select = list(set(dont_select_during_cahnce_loop + structure.already_minimum_section_story_indexes + structure.restrict_action_space()))
            else: 
                dont_select = list(set(dont_select_during_cahnce_loop + structure.already_minimum_section_story_indexes))

            if len(dont_select) >= len(structure.story_level_actions):
                done = True
                break
            
            dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else None
            action, _ = agent.choose_action(state, 
                                            dont_select, 
                                            dont_select_story_member_indexes,
                                            greedy=True)
            structure, reward, done, fail_name, fail_reason = env.step(structure, action)
            chances -= 1

        # get next state
        graph = structure.graph.clone()
        timestep += 1
        accumulated_reward += reward
        logger.info(f"timestep: {timestep}, accumulated_reward: {accumulated_reward}\n")
        action_list.append(action)
    
        # if can't select anymore, then stop
        if agent.restrict_action:
            dont_select = list(set(structure.already_minimum_section_story_indexes + structure.restrict_action_space()))
        else: 
            dont_select = structure.already_minimum_section_story_indexes

        if len(dont_select) >= len(structure.story_level_actions):
            done = True

    # generate final pisa ipt file
    if testing_structure:
        save_ipt_path = env.checkpoint_dir / f"final_design_testing_{original_chances}chance.ipt"
    elif taller_structure:
        save_ipt_path = env.checkpoint_dir / f"final_design_taller_{original_chances}chance.ipt"
    else:
        save_ipt_path = env.checkpoint_dir / f"final_design_random_{original_chances}chance.ipt"
    pisa._generate_analysis_ipt(original_structure, save_ipt_path, analysis="modal")

    # generate animation
    _frames_to_video(env.checkpoint_dir, save_dir, testing_structure, taller_structure, chances=original_chances)

    print(f"action list: {action_list[:-1]}")
    print(f"final design: {original_structure.story_level_sections}")
    print(f"material usage: {env.material_usage_record[-1]:5.2f} m3")
    print(f"reduced material: {np.sum(env.saved_material_record):5.2f} m3")  
