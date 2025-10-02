import os
import json
import torch
import numpy as np
from argparse import ArgumentParser
from NonlinearDynamicAnalysisSimulator.lstm import GraphLSTM 


def _read_args(folder):
    json_path = folder / "training_args.json"
    args_json = json.loads(open(json_path, 'r').read())
    args = ArgumentParser().parse_args()
    args_dict = vars(args)
    args_dict.update(args_json)
    return args


def _read_norm_dict(folder):
    norm_dict_path = folder / "norm_dict.json"
    norm_dict = json.loads(open(norm_dict_path, 'r').read())
    return norm_dict


def _init_model(folder, args, device):
    model_path = folder / "Models" / "model_Best.pt"
    model_kwargs = {"node_dim": 35, "edge_dim": 4, 
                    "gnn_num_layers": args.gnn_num_layers, "head_num": args.head_num,
                    "gnn_hidden_dim": args.gnn_hidden_dim, "latent_dim": args.latent_dim,
                    "graph_lstm_hidden_dim": args.graph_lstm_hidden_dim, "graph_lstm_num_layers": args.graph_lstm_num_layers,
                    "node_lstm_hidden_dim": args.node_lstm_hidden_dim, "node_lstm_num_layers": args.node_lstm_num_layers,
                    "ground_motion_dim": 20, "output_dim": 30, "device": device}
    model = GraphLSTM(**model_kwargs).to(device)
    print("--- Loading GraphLSTM from:", str(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model


def load_nonlinear_dynamic_analysis_simulator(graph_lstm_dir, device):
    args = _read_args(graph_lstm_dir)
    norm_dict = _read_norm_dict(graph_lstm_dir)
    nonlinear_dynamic_analysis_simulator = _init_model(graph_lstm_dir, args, device)
    return nonlinear_dynamic_analysis_simulator, norm_dict




def cut_gm(gm):
    # arias intensity
    # gm: [timestep, 10]
    gm = gm[:, 0]
    timestep = gm.shape[0]
    intensity = np.zeros_like(gm)

    # normalize gm (avoid very big intensity)
    gm = gm / np.max(np.abs(gm))

    # calculate intensity
    intensity[0] = gm[0] ** 2
    for i in range(1, len(gm)):
        intensity[i] = intensity[i-1] + gm[i] ** 2

    # normalize intensity to percentage
    intensity = intensity / np.max(intensity) * 100

    # get 5% and 95% timestep index
    threshold_start = 1
    threshold_end = 90
    index_start = np.argmin(np.abs(intensity - threshold_start))
    index_end = np.argmin(np.abs(intensity - threshold_end))

    # pad 5 second before and after
    pad = 5 * 20
    index_start = max(0, index_start - pad)
    index_end = min(timestep, index_end + pad)

    # plt.plot(gm)
    # plt.axvline(index_start, color='r', linewidth=3)
    # plt.axvline(index_end, color='r', linewidth=3)
    # plt.show()

    return index_start, index_end
    

freq = 200
five_second_data_num = freq * 5
original_max_timestep = 1300
cut_max_timestep = 20 * 25  # 20 Hz * 25 sec
def _read_ground_motion_text_from_folder(gm_folder):
    # there should be a XXX_FN.txt and a XXX_FP.txt
    files = [gm_folder / file for file in os.listdir(gm_folder)]
    assert len(files) == 2, "Both FN and FP files should exist"

    # read gm
    gm1 = np.loadtxt(files[0])[:, 1]
    gm2 = np.loadtxt(files[1])[:, 1]

    # padded_gm
    pad_gm1 = np.zeros(original_max_timestep * 10)
    pad_gm2 = np.zeros(original_max_timestep * 10)
    current_timestep = gm1.shape[0]
    pad_gm1[:current_timestep] = gm1
    pad_gm2[:current_timestep] = gm2

    # reshape to 20 Hz
    pad_gm1 = pad_gm1.reshape((-1, 10))  # shape: (1300, 10)
    pad_gm2 = pad_gm2.reshape((-1, 10))  # shape: (1300, 10)

    # remove starting and ending part of ground motion for efficiency
    start_index, end_index = cut_gm(pad_gm1)
    end_index = min(end_index, start_index + cut_max_timestep)
    remain_timestep = end_index - start_index
    cut_gm1 = np.zeros((cut_max_timestep, 10))
    cut_gm2 = np.zeros((cut_max_timestep, 10))
    cut_gm1[:remain_timestep, :] = pad_gm1[start_index:end_index, :]  # shape: (500, 10)
    cut_gm2[:remain_timestep, :] = pad_gm2[start_index:end_index, :]  # shape: (500, 10)

    # combine and convert to torch tensor
    gms = torch.from_numpy(np.concatenate([cut_gm1, cut_gm2], axis=1)).float()  # shape: (500, 10+10)

    # add a batch size dimension
    gms = gms.unsqueeze(0)  # shape: (1, 500, 10+10)

    return gms


def load_ground_motions(ground_motion_dir, ground_motion_number, nda_norm_dict):
    DBE_ground_motion_set = []
    MCE_ground_motion_set = []
    for gm_folder in os.listdir(ground_motion_dir)[:ground_motion_number]:
        print("---Loading ground motion:", gm_folder)
        gm_folder = ground_motion_dir / gm_folder
        ground_motions = _read_ground_motion_text_from_folder(gm_folder)  # shape: (1, 500, 10+10)
        ground_motions = (ground_motions - nda_norm_dict["ground_motion"][0]) / (nda_norm_dict["ground_motion"][1] - nda_norm_dict["ground_motion"][0])
        MCE_ground_motion_set.append(ground_motions)
        DBE_ground_motion_set.append(ground_motions * 3 / 4)

    return DBE_ground_motion_set, MCE_ground_motion_set

