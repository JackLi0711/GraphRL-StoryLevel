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
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from RL import agent, environment
from Structure import structure, pisa
from Structure.sections import beam_sections, column_sections


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




def _visualize_one_iteration(structure: structure.Structure, 
                             iteration: int, 
                             env: environment.Environment, 
                             accumulated_reward: float, 
                             q_values: torch.Tensor, 
                             save_fig_path: Path):
    # plot 3d
    fig = plt.figure(figsize=(20, 15), facecolor="w")

    # First plot q_values
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="w")
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

    infos = f"material usage: {env.material_usage_record[-1]:5.2f} m3\n" + f"reduced material: {saved_material:5.2f} m3\n"
    if env.scwb_driven_design:
        infos += f"reduced material(SCWB): {saved_material_SCWB:5.2f} m3"
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
