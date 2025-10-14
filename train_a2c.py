"""
Training script for A2C (Advantage Actor-Critic)

Author: Claude Code
Date: 2025-10-14
"""

import json
import torch
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("RL/")
sys.path.append("Structure/")
sys.path.append("Visualization")
sys.path.append("NonlinearDynamicAnalysisSimulator/")

from RL.pg import A2CAgent
from RL.pg.train_utils import train_episode, test_episode, plot_training_testing_curves, save_model
from RL import environment, record
from Visualization import plot, visualize
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # comment
    parser.add_argument("--comment", type=str, default="A2C with entropy annealing")

    # checkpoint
    parser.add_argument("--ckpt_dir", type=Path, default="./Results/A2C")
    parser.add_argument("--suffix", type=str, default="")

    # nonlinear dynamic analysis simulator
    parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
    parser.add_argument("--check_acceleration", action="store_true", default=False)
    parser.add_argument("--check_displacement", action="store_true", default=True)
    parser.add_argument("--graph_lstm_dir", type=Path, default=None)
    parser.add_argument("--ground_motion_dir", type=Path, default=None)
    parser.add_argument("--ground_motion_number", type=int, default=11)

    # structure
    parser.add_argument("--structure_shape", type=str, default="random", help="fixed, small_random, random")
    parser.add_argument("--add_structure_geometry", action="store_true", default=True)
    parser.add_argument("--add_response_features", action="store_true", default=False)
    parser.add_argument("--reward_type", type=str, default="material", help="material, acceleration, displacement, normalized, total, combined")

    # model
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)

    # A2C hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef_initial", type=float, default=0.01)
    parser.add_argument("--entropy_decay", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--accumulate_episodes", type=int, default=1)

    # training
    parser.add_argument("--num_epoch", type=int, default=1000, help="epoch == episode")
    parser.add_argument("--test_frequency", type=int, default=5)
    parser.add_argument("--test_runs", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=731)

    args = parser.parse_args()
    return args


def set_random_seed(SEED: int):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)


def get_loggings(ckpt_dir):
    logger = logging.getLogger(name='A2C-RL')
    logger.setLevel(level=logging.INFO)
    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # file handler
    file_handler = logging.FileHandler(ckpt_dir / "record.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main(args):
    # set random seed
    set_random_seed(args.random_seed)

    # set checkpoint directory
    if len(args.suffix) > 0:
        args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}__{args.suffix}'
    else:
        args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}'
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # set logger
    logger = get_loggings(args.ckpt_dir)
    logger.critical(args.ckpt_dir)
    logger.critical(args)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(device) if device == "cuda" else "CPU"
    logger.critical(f"Device: {device_name}")

    # setup nonlinear dynamic analysis simulator
    nda_simulator = None
    nda_norm_dict = None
    DBE_ground_motion_set = None
    MCE_ground_motion_set = None
    if args.do_nonlinear_dynamic_analysis:
        nda_simulator, nda_norm_dict = load_simulator.load_nonlinear_dynamic_analysis_simulator(args.graph_lstm_dir, device)
        DBE_ground_motion_set, MCE_ground_motion_set = load_simulator.load_ground_motions(args.ground_motion_dir, args.ground_motion_number, nda_norm_dict)

    # A2C Agent
    node_feature_dim = 8 if args.add_structure_geometry else 5
    edge_feature_dim = 13 if args.add_response_features else 11

    a2c_agent = A2CAgent(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        gamma=args.gamma,
        value_loss_coef=args.value_loss_coef,
        entropy_coef_initial=args.entropy_coef_initial,
        entropy_decay=args.entropy_decay,
        max_grad_norm=args.max_grad_norm,
        accumulate_episodes=args.accumulate_episodes,
        device=device,
        logger=logger
    )

    # Environment
    env = environment.Environment(
        structure_shape=args.structure_shape,
        add_structure_geometry=args.add_structure_geometry,
        add_response_features=args.add_response_features,
        reward_type=args.reward_type,
        scwb_driven_design=False,
        do_nonlinear_dynamic_analysis=args.do_nonlinear_dynamic_analysis,
        check_acceleration=args.check_acceleration,
        check_displacement=args.check_displacement,
        nda_simulator=nda_simulator,
        nda_norm_dict=nda_norm_dict,
        DBE_ground_motion_set=DBE_ground_motion_set,
        MCE_ground_motion_set=MCE_ground_motion_set,
        checkpoint_dir=args.ckpt_dir,
        logger=logger,
        device=device
    )

    # Record
    rec = record.Record()

    # Save args
    with open(args.ckpt_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    # Training loop
    logger.critical("Start training A2C...")
    for episode in range(args.num_epoch):
        # Training episode
        score, loss_dict = train_episode(a2c_agent, env, rec, logger)

        # Record training results
        rec.training_record["score"].append(score)
        if loss_dict:
            rec.learn_losses.append(loss_dict.get('total_loss', 0))

        logger.info(f"Episode {episode+1}/{args.num_epoch}, Score: {score:.4f}, "
                   f"Entropy Coef: {a2c_agent.get_entropy_coef():.6f}")
        if loss_dict:
            logger.info(f"  Policy Loss: {loss_dict.get('policy_loss', 0):.4f}, "
                       f"Value Loss: {loss_dict.get('value_loss', 0):.4f}, "
                       f"Entropy: {loss_dict.get('entropy', 0):.4f}")

        # Testing
        if (episode + 1) % args.test_frequency == 0:
            logger.critical(f"Testing at episode {episode+1}...")
            test_scores = []
            test_designs = []

            for run in range(args.test_runs):
                test_score, design_process = test_episode(a2c_agent, env, rec, logger)
                test_scores.append(test_score)
                test_designs.append(design_process)

            # Compute statistics
            mean_score = np.mean(test_scores)
            std_score = np.std(test_scores)

            # Record mean ± std
            rec.testing_record["score_mean"].append(mean_score)
            rec.testing_record["score_std"].append(std_score)

            # Find best design
            best_idx = np.argmax(test_scores)
            best_design = test_designs[best_idx]
            rec.testing_record["best_designs"].append(best_design)

            logger.critical(f"Testing Score: {mean_score:.4f} ± {std_score:.4f}")

            # Output and plot
            rec.output(args.ckpt_dir)
            plot_training_testing_curves(rec, args.ckpt_dir, args.test_frequency)

            # Save model
            save_model(a2c_agent, args.ckpt_dir, episode)

    logger.critical("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
