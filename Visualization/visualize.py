import os
import glob
import time
import torch
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

from PIL import Image
from typing import Union
from pathlib import Path
from copy import deepcopy
from logging import Logger
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from RL import environment, agent_DQN, agent_PPO, agent_OC, agent_OC_m3, new_strategy, rollout
from Structure import pisa, check, opensees
from Structure.structure import Structure
from Structure.sections import beam_sections, column_sections, YIELDING_STRESS


@torch.no_grad()
def visualize_edge_embedding(agent, env, logger, ckpt_dir):
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


@torch.no_grad()
def visualize_option_policy(agent: agent_OC.OptionCriticAgent, 
                          env: environment.Environment, 
                          logger: Logger, 
                          testing_structure: bool=False,
                          taller_structure: bool=False,
                          initial_design: list[int]=None, 
                          option_idx: int=0):
    logger.info(f"Visualizing option embedding (TSNE)......")
    
    # get testing structure & graph
    device = agent.device
    if testing_structure:
        structure = env.reset(testing=True, initial_design=initial_design)
        save_name = f"testing_option{option_idx}"
    elif taller_structure:
        structure = env.reset(taller=True, initial_design=initial_design)
        save_name = f"taller_option{option_idx}"
    else:
        structure = env.reset(initial_design=initial_design)
        save_name = f"x{structure.x_span_num}z{structure.z_span_num}y{structure.story_num}_option{option_idx}"
    graph = structure.graph.clone()

    done = None
    accumulated_reward = 0
    timestep = 0
    action_list = []
    option_embedding_list = []
    while not done:
        original_structure = deepcopy(structure)
        graph = graph.to(device)
        state = agent.gnn(graph.x, graph.edge_index, graph.edge_attr, None, structure.aux["story_batch"].to(device), None)

        # select action and update structure
        print(f"story level sections: {structure.story_level_sections}")
        print(f"original minimum: {structure.already_minimum_section_story_indexes}")
        print(f"restrict actions: {structure.restrict_action_space()}")
        
        dont_select_story_member_indexes = structure.restrict_action_space() if agent.restrict_action else []
        infeasible_actions = list(set(structure.already_minimum_section_story_indexes + dont_select_story_member_indexes))
        infeasible_mask = torch.tensor([True if i in infeasible_actions else False for i in range(state.shape[0])], dtype=torch.bool, device=device)
        masks = infeasible_mask.unsqueeze(0).repeat(agent.num_options, 1)  # (num_actions) --> (num_options, num_actions)
        
        logits, value = agent.option_critic_network.forward(state)  # (num_options, num_actions), (1,)
        masked_logits = logits.masked_fill(masks, float('-inf'))  # set infeasible actions' logits to -inf
        probs = torch.nn.functional.softmax(masked_logits, dim=-1)  # (num_options, num_actions)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (num_options, num_actions), use original logits to avoid -inf values, which will cause issue in t-SNE calculation
        option_embedding_list.append(log_probs.detach().cpu())

        # update action
        action = probs[option_idx].argmax().item()
        structure, reward, done, fail_name, fail_reason = env.step(structure, action)

        # get next state
        graph = structure.graph.clone()
        timestep += 1
        accumulated_reward += reward
        logger.info(f"timestep: {timestep}, action: {action}, reward: {reward}, accumulated_reward: {accumulated_reward}\n")
        action_list.append(action)
    
        # if can't select anymore, then stop
        if agent.restrict_action:
            dont_select = list(set(structure.already_minimum_section_story_indexes + structure.restrict_action_space()))
        else: 
            dont_select = structure.already_minimum_section_story_indexes
        if len(dont_select) >= len(structure.story_level_actions):
            done = True

    option_embeddings = torch.stack(option_embedding_list, dim=0)  # (num_timesteps, num_options, num_actions)
    T, K, A = option_embeddings.shape
    all_embeddings = option_embeddings.reshape(T*K, A).numpy()

    tsne = TSNE(n_components=2, init='random', perplexity=10, n_iter=1000)
    embed_2d = tsne.fit_transform(all_embeddings)  # (T*K, 2)
    embed_2d = embed_2d.reshape(T, K, 2)

    # Plot t-SNE scatter plot between options
    plt.figure(figsize=(8, 6), facecolor="w")
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'brown', 'pink', 'gray'][:K]
    for k in range(K):
        for t in range(T):
            alpha = 0.3 + 0.7 * (t / T)
            plt.scatter(embed_2d[t, k, 0], embed_2d[t, k, 1], c=colors[k], s=50, alpha=alpha, label=f'Option {k}' if t == T-1 else "")
    plt.xlabel('TSNE 1', fontsize=14)
    plt.ylabel('TSNE 2', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.title('t-SNE of Option Policies (Log Probability)', fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(env.checkpoint_dir/f"option_policy_tsne_{save_name}.png", dpi=100, bbox_inches='tight')
    plt.close()

    # Plot cosine similarity heatmap between options
    cos_sim_array = np.zeros((K, K))
    for i in range(K): 
        for j in range(K): 
            cos_sim_array[i, j] = torch.nn.functional.cosine_similarity(option_embeddings[:, i, :], option_embeddings[:, j, :], dim=-1).mean().item()
    plt.figure(figsize=(8, 6), facecolor="w")
    plt.imshow(cos_sim_array, cmap='viridis', vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            plt.text(j, i, f"{cos_sim_array[i, j]:.3f}", ha='center', va='center', color='white' if cos_sim_array[i, j] < 0.5 else 'black')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(K), [f'Option {k}' for k in range(K)], rotation=45)
    plt.yticks(range(K), [f'Option {k}' for k in range(K)])
    plt.title('Cosine Similarity Between Option Policies (Log Probability)', fontsize=16)
    plt.tight_layout()
    plt.savefig(env.checkpoint_dir/f"option_policy_cossim_{save_name}.png", dpi=100, bbox_inches='tight')
    plt.close()




def _visualize_one_iteration(structure: Structure, 
                             iteration: int, 
                             env: environment.Environment, 
                             vis_values: torch.Tensor, 
                             save_fig_path: Path):
    # plot 3d
    fig = plt.figure(figsize=(20, 15), facecolor="w")

    # First plot q_values
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="w")
    ax.set_axis_off()

    vis_values_member = [0 for _ in range(structure.member_number)]
    for i in range(len(vis_values)):
        for member_index in structure.story_level_actions[i]:
            vis_values_member[member_index] = vis_values[i]

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
        # member visualization value
        member_index = int(member_name[1:]) - 1
        value = 0 if member_index in structure.already_minimum_section_story_indexes else vis_values_member[member_index]
        color = cmap(value)[:3]
        edges_color.append(color)

    for edge_i in range(len(edges_coord)):
        x1, y1, z1, x2, y2, z2 = edges_coord[edge_i]
        xx = [x1, x2]
        yy = [y1, y2]
        zz = [z1, z2]
        color = edges_color[edge_i]
        # plot a border on a line: https://stackoverflow.com/questions/12729529/can-i-give-a-border-outline-to-a-line-in-matplotlib-plot-function
        ax.plot(xx, zz, yy, c=(color), linewidth=5, path_effects=[pe.Stroke(linewidth=9, foreground='black'), pe.Normal()])

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    sm = matplotlib.cm.ScalarMappable(cmap='Greys', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Value', fontsize=24)
    ax.set_title(f"Visualization of member values", fontsize=30)


    # Then plot beam-column sections
    ax = fig.add_subplot(1, 2, 2, projection="3d", facecolor="w")
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

    infos = f"Material usage: {env.material_usage_record[-1]:5.2f} m3\n" + f"Reduced material: {saved_material:5.2f} m3\n"
    if env.scwb_driven_design:
        infos += f"Reduced material(SCWB): {saved_material_SCWB:5.2f} m3"
    title = f"Reward: {env.reward_type}\n" + f"Iteration: {iteration:4d}\n" + infos

    ax.set_title(title, fontsize=30)
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


def visualize_design_process(agent: Union[agent_DQN.DeepQAgent, agent_PPO.PPOAgent, agent_OC.OptionCriticAgent], 
                             env: environment.Environment, 
                             logger: Logger, 
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
        save_dir = env.checkpoint_dir / f"design_process_random_{original_chances}chance"
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
        infeasible_actions = list(set(structure.already_minimum_section_story_indexes + dont_select_story_member_indexes))
        if isinstance(agent, agent_DQN.DeepQAgent):
            action, q_val = agent.choose_action(state, infeasible_actions, greedy=True)
            q_values = agent.online_q_network(state).squeeze().detach().cpu().numpy()
            vis_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))  # noamalize q_values to [0, 1]
        elif isinstance(agent, agent_PPO.PPOAgent):
            logits, value, entropy, action, log_prob = agent.choose_action(state, infeasible_actions, greedy=True)
            vis_values = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()  # use action probabilities as vis values
        elif isinstance(agent, agent_OC.OptionCriticAgent):
            option_idx = 0
            gjsd, logits, value, entropy, action, log_prob = agent.choose_action(state, infeasible_actions, option_idx, greedy=True)
            vis_values = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()  # use action probabilities as vis values
        else:
            raise ValueError("agent type not supported in visualization.")

        # visualize
        vis_path = save_dir / f"{timestep}.png"
        _visualize_one_iteration(structure, timestep, env, vis_values, vis_path)

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
            
            if isinstance(agent, agent_DQN.DeepQAgent):
                action, q_val = agent.choose_action(state, dont_select, greedy=True)
            elif isinstance(agent, agent_PPO.PPOAgent):
                logits, value, entropy, action, log_prob = agent.choose_action(state, dont_select, greedy=True)
            elif isinstance(agent, agent_OC.OptionCriticAgent):
                option_idx = 4
                gjsd, logits, value, entropy, action, log_prob = agent.choose_action(state, dont_select, option_idx, greedy=True)
            else:
                raise ValueError("agent type not supported in visualization.")
            structure, reward, done, fail_name, fail_reason = env.step(structure, action)
            chances -= 1

        # get next state
        graph = structure.graph.clone()
        timestep += 1
        accumulated_reward += reward
        logger.info(f"timestep: {timestep}, action: {action}, reward: {reward}, accumulated_reward: {accumulated_reward}\n")
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




def get_axial_moment_interaction_ratio(structure: Structure, response: opensees.NewResponse) -> np.ndarray:
    """
    Get axial-moment-interaction ratio given a specific structure and response.
    - [鋼構規範(LRFD) 8.2 對稱構材承受彎矩及軸力之作用](https://www.nlma.gov.tw/filesys/file/chinese/publication/law/law/3495-8.pdf)
    """
    PHI, PHI_b = 0.85, 0.90
    
    # demand
    axial_force = np.array(list(response.member_response["axial"].values()))
    Pu = np.abs(axial_force)
    Mux = np.abs(np.array(list(response.member_response["momentZ"].values())))
    Muy = np.abs(np.array(list(response.member_response["momentY"].values())))

    # capacity
    A = np.array(list(structure.member_A_dict.values()))  # mm2
    Fcr = np.array(list(structure.member_Fcr_dict.values()))  # kN/mm2
    Puc = Fcr * A  # kN
    Put = YIELDING_STRESS * A  # kN

    Pn = np.zeros_like(Pu)
    Pn += Put * (axial_force > 0)  # tension condition
    Pn += Puc * (axial_force <= 0)  # compression condition
    phi_Pn = PHI * Pn

    Mnx = np.array(list(structure.member_Mnx_dict.values()))
    Mny = np.array(list(structure.member_Mny_dict.values()))
    phi_Mnx = PHI_b * Mnx
    phi_Mny = PHI_b * Mny
    
    # interaction ratio
    case_1 = ((Pu / phi_Pn) >= 0.2)
    ratio_1 = ((Pu / phi_Pn) + 8/9 * (Mux / phi_Mnx + Muy / phi_Mny)) * case_1
    case_2 = ((Pu / phi_Pn) < 0.2)
    ratio_2 = ((Pu / (2 * phi_Pn)) + (Mux / phi_Mnx + Muy / phi_Mny)) * case_2
    ratio = ratio_1 + ratio_2

    return ratio


def process_pmm_ratio(structure: Structure, analysis_dir: Path):
    load_cases, static_responses = check.get_response(structure, analysis_dir)
    stress_ratios = np.zeros((structure.member_number, len(load_cases)))
    pmm_ratios = np.zeros((structure.member_number, len(load_cases)))
    for i, (load_case, response) in enumerate(zip(load_cases, static_responses)):
        beam_com_ratios = check.get_ratio_beam_compression_strength(structure, response)
        beam_ten_ratios = check.get_ratio_beam_tension_strength(structure, response)
        col_com_ratios = check.get_ratio_column_compression_strength(structure, response)
        col_ten_ratios = check.get_ratio_column_tension_strength(structure, response)
        stress_ratios[structure.member_beam_index_list, i] = beam_com_ratios + beam_ten_ratios
        stress_ratios[structure.member_column_index_list, i] = col_com_ratios + col_ten_ratios
        
        beam_pmm_ratios = check.get_ratio_beam_axial_moment(structure, response)
        pmm_ratios[structure.member_beam_index_list, i] = beam_pmm_ratios
        pmm_ratios[structure.member_column_index_list, i] = col_com_ratios + col_ten_ratios

    pmm_ratios_max = pmm_ratios.max(axis=1)
    print(f"max beam PMM ratio: {pmm_ratios_max[structure.member_beam_index_list].max():.3f}, limit: {check.BEAM_AXIAL_MOMENT_LIMIT:.3f}")
    print(f"max colu PMM ratio: {pmm_ratios_max[structure.member_column_index_list].max():.3f}, limit: {check.PHI_C:.3f}")

    pmm_ratio_story_member_group_infos = np.zeros((len(structure.story_level_sections), 2))  # 2: max/mean of story member group
    for i, story_member_indices in enumerate(structure.story_level_actions):
        pmm_ratios_story_member_group = pmm_ratios_max[story_member_indices]
        pmm_ratio_story_member_group_infos[i, 0] = pmm_ratios_story_member_group.max()
        pmm_ratio_story_member_group_infos[i, 1] = pmm_ratios_story_member_group.mean()
    
    return pmm_ratio_story_member_group_infos


def get_inference_info(agent: Union[agent_DQN.DeepQAgent, agent_PPO.PPOAgent, agent_OC.OptionCriticAgent, agent_OC_m3.OptionCriticAgent], 
                       env: environment.Environment, 
                       geo_info: list[int],
                       initial_design: list[int] = None, 
                       infos: list[str] = None):
    x_span_num, z_span_num, story_num, x_span_len, z_span_len, story_height = geo_info
    geo_name = f"x{x_span_num}_z{z_span_num}_y{story_num}_{x_span_len}_{z_span_len}_{story_height}"
    save_dir = env.checkpoint_dir / "inference" / geo_name
    save_dir.mkdir(parents=True, exist_ok=True)
    if "visualization" in infos:
        frame_dir = save_dir / "design_process"
        frame_dir.mkdir(parents=True, exist_ok=True)

    structure_kwargs = {
        'x_span_num': x_span_num,
        'x_span_lens': [x_span_len] * x_span_num,
        'z_span_num': z_span_num,
        'z_span_lens': [z_span_len] * z_span_num,
        'story_num': story_num,
        'story_height': story_height,
        'story_level_sections': None if initial_design is None else initial_design,
        'add_structure_geometry': env.add_structure_geometry,
        'add_response_features': env.add_response_features,
        'do_nonlinear_dynamic_analysis': env.do_nonlinear_dynamic_analysis,
        'nda_norm_dict': env.nda_norm_dict,
        'analysis_dir': env.checkpoint_dir/"Modal_Analysis"
    }
    structure = Structure(**structure_kwargs)
    print(structure)
    if env.scwb_driven_design:
        new_strategy.strong_column_weak_beam_driven_update(structure, env.code_analysis_dir)
    env.init_records(structure)
    
    timestep = 0
    accumulated_reward = 0
    action_list = []
    times = [[], []]  # agent.choose_action(), env.step()
    pmm_ratio_infos = [[], []]  # max and mean of PMM ratio

    def before_step(step: rollout.DesignStep, structure_before_step: Structure):
        print(f"used time for agent.choose_action(): {step.choose_time:.3f}s")
        # visualize
        if "visualization" in infos:
            vis_path = frame_dir / f"{step.index}.png"
            _visualize_one_iteration(structure_before_step, step.index, env, step.vis_values, vis_path)

    for step in rollout.design_steps(agent, env, structure, before_step=before_step):
        original_structure = step.structure_before
        structure, action, reward = step.structure_after, step.action, step.reward
        times[0].append(step.choose_time)
        member_category = structure.story_level_categories[action]
        update_story = (action % structure.story_num) + 1
        action_info = f"{update_story}F_{member_category}"        
        times[1].append(step.step_time)
        print(f"used time for env.step(): {step.step_time:.3f}s")

        # get PMM ratio
        if "pmm_ratio" in infos:
            pmm_ratio_story_member_group_infos = process_pmm_ratio(structure, env.checkpoint_dir)
            member_max_group_max, member_mean_group_max = np.max(pmm_ratio_story_member_group_infos, axis=0)
            member_max_group_min, member_mean_group_min = np.min(pmm_ratio_story_member_group_infos, axis=0)
            member_max_group_med, member_mean_group_med = np.median(pmm_ratio_story_member_group_infos, axis=0)
            member_max_choose_max, member_mean_choose_mean = pmm_ratio_story_member_group_infos[action]
            # member_max_choose_max_rank = np.sum(pmm_ratio_story_member_group_infos[:, 0] > member_max_choose_max) + 1
            pmm_ratio_infos[0].append([member_max_group_max, member_max_group_min, member_max_group_med, member_max_choose_max])
            pmm_ratio_infos[1].append([member_mean_group_max, member_mean_group_min, member_mean_group_med, member_mean_choose_mean])

        timestep += 1
        accumulated_reward += reward
        action_list.append(action)
        print(f"timestep: {timestep:3d}, action: {action:3d} [{action_info}], reward: {reward:6.3f}, accumulated_reward: {accumulated_reward:6.3f}\n")

    print(original_structure)
    print(f"action list: {action_list[:-1]}")
    print(f"final design: {original_structure.story_level_sections}")
    print(f"material usage: {env.material_usage_record[-1]:5.2f} m3")
    print(f"reduced material: {np.sum(env.saved_material_record):5.2f} m3") 

    inference_times = {'agent_inference': times[0], 'env_step': times[1]}
    if "time" in infos:
        print("\ninference times:")
        for category, times_list in inference_times.items():
            print(f"  {category}: {np.mean(times_list):.3f} ± {np.std(times_list):.3f} s")

    if "pmm_ratio" in infos:
        pmm_ratio_array = np.array(pmm_ratio_infos)  # (2, num_timesteps, 4)
        max_pmm_ratios, mean_pmm_ratios = pmm_ratio_array[0], pmm_ratio_array[1]  # max/mean of member within the same group
        timesteps = np.arange(len(max_pmm_ratios))
        stats = ["max", "min", "med"]
        plt.figure(figsize=(12, 6))
        for i, stat in enumerate(stats):
            plt.plot(timesteps, max_pmm_ratios[:, i], label=f"{stat}", color="red", linestyle="--", alpha=0.5)
        plt.plot(timesteps, max_pmm_ratios[:, 3], label=f"chosen", color="red", linewidth=2)
        # for i, stat in enumerate(stats):
        #     plt.plot(timesteps, mean_pmm_ratios[:, i], label=f"member: mean, group: {stat}", color="blue", alpha=0.5)
        # plt.plot(timesteps, mean_pmm_ratios[:, 3], label=f"member: mean, chosen group", color="blue", linewidth=2)

        plt.xlabel("Timestep", fontsize=14)
        plt.ylabel("PMM Ratio", fontsize=14)
        plt.title("PMM Ratio of Story Member Groups During Design Process", fontsize=16)
        plt.legend(loc="best", fontsize=14)
        plt.grid()
        plt.tight_layout()
        save_path = save_dir / "pmm_ratio.png"
        plt.savefig(save_path, bbox_inches="tight")

    if "visualization" in infos:
        # generate final pisa ipt file
        save_ipt_path = save_dir / f"final_design.ipt"
        pisa._generate_analysis_ipt(original_structure, save_ipt_path, analysis="modal")

        # generate animation
        frame_names = [image for image in glob.glob(f"{frame_dir}/*.png")]
        frame_names = [str(frame_dir / f"{i}.png") for i in range(0, len(frame_names))]
        print(f"frame names length: {len(frame_names)}")
        frames = [Image.open(name) for name in frame_names]
        last_frame = frames[-1]
        frames += [last_frame] * 5
        
        # frame_duration_ms = 1000 if not short else 200
        frame_duration_ms = 200
        frame_one = frames[0]
        save_animation_path = save_dir / "design_animation.gif"            
        frame_one.save(save_animation_path, format="GIF", append_images=frames, save_all=True, duration=frame_duration_ms, loop=0)
