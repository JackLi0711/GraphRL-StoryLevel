"""
Check whether a structure pass the criteria about the drift ratio
and plastic hinge behavior under nonlinear dynamic analysis (NDA) that we set. 
"""
import time
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple
from Structure.structure import Structure
from torch_geometric.loader import DataLoader


def check(structure: Structure,
          nda_simulator: torch.nn.Module,
          DBE_ground_motion_set: List[torch.Tensor],
          MCE_ground_motion_set: List[torch.Tensor],
          check_acceleration: bool,
          check_displacement: bool,
          nda_norm_dict: Dict[str, float],
          auxiliary_values: Dict[str, float],
          device: torch.device,
          logger: logging.Logger=None) -> Tuple[bool, str]:
    
    # inference on model to get the response
    # start = time.time()
    mce_nda_responses = _generate_nda_response(structure, nda_simulator, MCE_ground_motion_set, device)
    # end = time.time()
    # print(f"GraphLSTM uses: {end-start:3.3f} sec")

    # check plastic hinge location
    if not _check_nda_plastic_hinge_location_pass(structure, mce_nda_responses): return False, "nda_hinge"

    # check drift ratio
    if check_displacement:
        if not _check_nda_drift_ratio_pass(structure, mce_nda_responses, nda_norm_dict): return False, "nda_drift"
    
    # check maximum acceleration
    if check_acceleration:
        #dbe_nda_responses = _generate_nda_response(structure, nda_simulator, DBE_ground_motion_set, device)
        if not _check_nda_acceleration_pass(structure, MCE_ground_motion_set, mce_nda_responses, nda_norm_dict, auxiliary_values, logger): return False, "nda_acceleration"

    return True, None




'''
# run 3 ground motions, 1 by 1: 7-8 seconds
# run 3 ground motions, 3 in a batch: 5 seconds
# run 11 ground motions, 1 by 1: 17 seconds
# run 11 ground motions, 11 in a batch: 9 seconds
'''
def _generate_nda_response(structure: Structure,
                           nda_simulator: torch.nn.Module,
                           ground_motion_set: List[torch.Tensor],
                           device: torch.device,) -> np.array:
    
    # feed 11 gms as mini-batch
    gm_num = len(ground_motion_set)
    ground_motions = torch.cat(ground_motion_set, dim=0).to(device)
    timesteps = ground_motions.shape[1]  # ground_motions.shape: (11, 500, 10+10)

    # graph settings for foward propagation
    graph = structure.nda_graph.to(device)
    node_num = graph.x.shape[0]

    # turn into batch
    graph_loader = DataLoader([graph for _ in range(gm_num)], batch_size=gm_num, shuffle=None)
    graph_batch = next(iter(graph_loader))
    sampled_index_batch = [None for _ in range(gm_num)]

    with torch.no_grad():
        responses, _ = nda_simulator(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, 
                                     graph_batch.batch, graph_batch.ptr, sampled_index_batch, 
                                     ground_motions, sample_node=False)
        responses = responses.cpu().numpy()

    # reshape response from [batch_node, timesteps, output_dim] to [bacth_size, node_num, timesteps, output_dim]
    responses = responses.reshape(gm_num, node_num, timesteps, -1)

    return responses




NDA_DRIFT_RAIO_LIMIT = 0.03
def _check_nda_drift_ratio_pass(structure: Structure,
                                responses: np.array,
                                nda_norm_dict: Dict) -> bool:
    
    # average the response of all 11 ground motions --> shape: (node_num, timesteps, output_dim)
    response = np.mean(responses, axis=0)

    # denormalize the displacement
    response[:, :, 4:6] *= nda_norm_dict["disp"]

    # node_disp: [node_num, timesteps] 
    node_disp_x = response[:, :, 4]
    node_disp_z = response[:, :, 5]
    bottom_node_disp_x = node_disp_x[structure.bottom_node_index_list]
    bottom_node_disp_z = node_disp_z[structure.bottom_node_index_list]
    member_length = structure.bottom_member_length_array    # mm
    member_length = member_length[:, np.newaxis]
    
    # drift ratio
    timeseries_length = node_disp_x.shape[-1]
    member_length = np.tile(member_length, timeseries_length)
    drift_ratio_timeseries_x = np.abs((node_disp_x - bottom_node_disp_x) / member_length)
    drift_ratio_timeseries_z = np.abs((node_disp_z - bottom_node_disp_z) / member_length)
    max_drift_x = np.max(drift_ratio_timeseries_x)
    max_drift_z = np.max(drift_ratio_timeseries_z)
    # print("max_drift_x:", max_drift_x)
    # print("max_drift_z:", max_drift_z)

    # check if drift ratio is over than threshold
    if max_drift_x > NDA_DRIFT_RAIO_LIMIT or max_drift_z > NDA_DRIFT_RAIO_LIMIT:
        print("Nonlinear dynamic analysis drift ratio didn't pass!")
        return False
    return True


Myield_yn_index = 28
Myield_yp_index = 30
output_yn_index = 8
output_yp_index = 9
yield_factor = 0.90
def _check_nda_plastic_hinge_location_pass(structure: Structure,
                                           responses: np.array) -> bool:

    # check: plastic hinges can only be on 1F column bottom and beams
    # procedure: checking only columns are enough

    total_case = responses.shape[0]
    minimum_pass_case = total_case // 2 + 1
    pass_case = 0

    for response in responses:
        hinge_yn = (np.max(np.abs(response[structure.hinge_node_yn, :, output_yn_index]), axis=1)[0] >= yield_factor * structure.nda_graph.x[structure.hinge_node_yn, Myield_yn_index].cpu().numpy())
        hinge_yp = (np.max(np.abs(response[structure.hinge_node_yp, :, output_yp_index]), axis=1)[0] >= yield_factor * structure.nda_graph.x[structure.hinge_node_yp, Myield_yp_index].cpu().numpy())
        if np.any(hinge_yn) or np.any(hinge_yp):
            continue
        pass_case += 1
    
    # print(f"NDA_plastic_hinge, total_case: {total_case}, pass_case: {pass_case}")
    if pass_case < minimum_pass_case:
        print("Nonlinear dynamic analysis plastic hinge location didn't pass!")
        return False
    return True


# calculate the factor for the force amplification (a function of height)
def _Hf(current_height, total_height, first_mode_period):
    """
    ASCE 13.3.1.1 Amplification with Height, Hf
    """
    z = current_height
    h = total_height
    Ta = first_mode_period
    a1 = min(2.5, 1 / Ta)
    a2 = max(0, 1 - (0.4 / Ta) ** 2)

    # US ASCE 13.3.1.1
    Hf = 1 + a1 * (z / h) + a2 * (z / h) ** 10
    #Hf = 1 + 2.5 * (z / h)

    # TW 耐震設計規範 4.2
    #Hf = 1 + 2 * (z / h)

    return Hf


SDS = 0.6   # 表2.6
I = 1.0     # 2.8 第四類建築物其他一般建築物，I=1.0
def _check_nda_acceleration_pass(structure: Structure,
                                 MCE_ground_motion_set:List[torch.Tensor],
                                 responses: np.array,
                                 nda_norm_dict: Dict[str, float],
                                 auxiliary_values: Dict[str, float],
                                 logger: logging.Logger) -> bool:
    """
    耐震設計規範4.2
    Check if the maximum accelerations in each story are within the limit.

    For story i, the maximum acceleration is a_i = 0.4 * S_DS * I_p * H_f(i) / Fu
    The calculated maximum acceleration is in (g) unit

    responses: [bacth_size(11), node_num, timesteps(500), output_dim(18)]
    """

    # find the extreme acceleration in the time history for each node
    # ground_motion.shape:        (1, 500, 10+10) | (*, timestep, *)
    # ground_accelerations.shape: (11, 1, 500)    | (gm_num, *, timestep)
    ground_accelerations_x = torch.cat([ground_motion[:, :, 0].unsqueeze(0) for ground_motion in MCE_ground_motion_set], dim=0)
    ground_accelerations_z = torch.cat([ground_motion[:, :, 10].unsqueeze(0) for ground_motion in MCE_ground_motion_set], dim=0)
    
    node_num = responses.shape[1]
    # ground_acceleration.shape: (11, node_num, 500)
    ground_accelerations_x = ground_accelerations_x.expand(-1, node_num, -1).numpy()
    ground_accelerations_z = ground_accelerations_z.expand(-1, node_num, -1).numpy()

    # time_history_acc: shape = (batch_size, node_num, timestep)
    time_history_acc_x = responses[:, :, :, 0] #+ ground_accelerations_x
    time_history_acc_z = responses[:, :, :, 1] #+ ground_accelerations_z

    # node_acc: shape = (batch_size, node_num)
    node_acc_x = np.max(np.abs(time_history_acc_x), axis=2) * nda_norm_dict["acc"]
    node_acc_z = np.max(np.abs(time_history_acc_z), axis=2) * nda_norm_dict["acc"]

    for story in range(1, structure.story_num+1):
    # for story in range(structure.story_num, 0, -1):
        current_height = structure.y_grid[story]
        total_height = structure.y_grid[-1]
        story_name = str(story) + "F"

        target_node_index = structure.story_nodes[story_name]

        # the maximum story acceleration for each earthquakes
        story_acc_x = np.max(np.abs(node_acc_x[:, target_node_index]), axis=1)  # [batch_size]
        story_acc_z = np.max(np.abs(node_acc_z[:, target_node_index]), axis=1)  # [batch_size]

        # the averaged maximum story acceleration for all earthquakes
        averaged_story_acc_x = np.mean(story_acc_x)
        averaged_story_acc_z = np.mean(story_acc_z)

        # calculate the maximum acceleration
        Hf = _Hf(current_height, total_height, auxiliary_values["first_mode_period"])
        a_max_g = 0.4 * SDS * I * Hf #/ auxiliary_values["Fu"]

        # convert a_max_g from g to mm/s2
        a_max = a_max_g * 9800  # mm/s2

        #print(f"story: {story}F, Hf:", Hf)
        #print(f"story: {story}F, Fu:", auxiliary_values["Fu"])
        #print(f"story: {story}F, story_acc_x:", story_acc_x)
        #print(f"story: {story}F, story_acc_z:", story_acc_z)
        #print(f"story: {story}F, averaged_story_acc_x:", averaged_story_acc_x)
        #print(f"story: {story}F, averaged_story_acc_z:", averaged_story_acc_z)
        #print(f"story: {story}F, maximum acceleration:", a_max)

        if averaged_story_acc_x > a_max or averaged_story_acc_z > a_max:
            logger.info(f"NDA accelaration didn't pass! {story}F, Acc(x): {averaged_story_acc_x/9800:.3f}, Acc(z): {averaged_story_acc_z/9800:.3f}, limit: {a_max/9800:.3f}")
            return False

    return True




def record_acc(structure: Structure,
               nda_simulator: torch.nn.Module,
               MCE_ground_motion_set: List[torch.Tensor],
               nda_norm_dict: Dict[str, float],
               device: torch.device,
               acc_record: Dict[str, list],
               logger: logging.Logger,) -> Dict[str, list]:
    """
    Record the maximum node acceleration for each story in both two directions
    (take average of all selcted ground motions)
    """
    mce_nda_responses = _generate_nda_response(structure, nda_simulator, MCE_ground_motion_set, device)
    # time_history_acc: shape = (batch_size, node_num, timestep)
    time_history_acc_x = mce_nda_responses[:, :, :, 0]  # Abs. Acc.
    time_history_acc_z = mce_nda_responses[:, :, :, 1]  # Abs. Acc.

    # node_acc: shape = (batch_size, node_num, timestep)
    node_acc_x = np.abs(time_history_acc_x * nda_norm_dict["acc"])
    node_acc_z = np.abs(time_history_acc_z * nda_norm_dict["acc"])

    averaged_story_acc_x_list = []
    averaged_story_acc_z_list = []
    for story in range(1, structure.story_num+1):

        story_name = str(story) + "F"

        # node which locates at the center of mass of the slab
        target_node_index = structure.story_mass_center_nodes[story_name]
        # all nodes in the story
        index = structure.story_nodes[story_name]

        # 1. find averaged story acceleration for each earthquake and timestep
        # (batch_size, node_num, timestep) --> (batch_size, timestep)
        averaged_story_acc_x = np.mean(node_acc_x[:, target_node_index, :], axis=1)
        averaged_story_acc_z = np.mean(node_acc_z[:, target_node_index, :], axis=1)

        # 2. find all-timestep maximum averaged story acceleration for each earthquake
        # (batch_size, timestep) --> (batch_size)
        max_story_acc_x = np.max(averaged_story_acc_x, axis=1)
        max_story_acc_z = np.max(averaged_story_acc_z, axis=1)

        # 3. take average of maximum averaged story acceleration for all earthquakes
        # (batch_size) --> 1
        gm_mean_max_story_acc_x = np.mean(max_story_acc_x) / 9810  # mm/s2 --> g
        gm_mean_max_story_acc_z = np.mean(max_story_acc_z) / 9810  # mm/s2 --> g

        averaged_story_acc_x_list.append(gm_mean_max_story_acc_x)  
        averaged_story_acc_z_list.append(gm_mean_max_story_acc_z)

        # logger.info(f"{story}F ceiling, Acc(x): {gm_mean_max_story_acc_x:.3f}, Acc(z): {gm_mean_max_story_acc_z:.3f}")

    acc_record['X-dir'].append(averaged_story_acc_x_list)
    acc_record['Z-dir'].append(averaged_story_acc_z_list)

    return acc_record