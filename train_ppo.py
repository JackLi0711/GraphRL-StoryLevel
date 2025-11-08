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

from RL.pg import PPOAgent
from RL.pg.train_utils import train_episode, test_episode, plot_training_testing_curves, save_model, plot_loss_curves, plot_gradient_norms
from RL import environment, record
from Visualization import plot, visualize
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # comment
    parser.add_argument("--comment", type=str, default="PPO with clipped surrogate objective and entropy annealing")

    # checkpoint
    parser.add_argument("--ckpt_dir", type=Path, default="./Results/PPO")
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

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_epsilon", type=float, default=0.3)
    parser.add_argument("--policy_loss_coef", type=float, default=10.0)
    parser.add_argument("--value_loss_coef", type=float, default=0.3)
    parser.add_argument("--entropy_coef_initial", type=float, default=0.01)
    parser.add_argument("--entropy_decay", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--accumulate_episodes", type=int, default=5)
    parser.add_argument("--state_gnn_lr_multiplier", type=float, default=3, help="StateGNN learning rate multiplier")
    # PPO enhancements
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--minibatch_size", type=int, default=64)
    # Failure penalty (alpha)
    parser.add_argument("--failure_penalty_ratio", type=float, default=0.2)

    # training
    parser.add_argument("--num_epoch", type=int, default=2000, help="epoch == episode")
    parser.add_argument("--test_frequency", type=int, default=10)
    parser.add_argument("--test_runs", type=int, default=1)
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
    logger = logging.getLogger(name='PPO-RL')
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

    # PPO Agent
    node_feature_dim = 8 if args.add_structure_geometry else 5
    # PPO uses extended edge features with member type and floor information
    edge_feature_dim = 18 if args.add_response_features else 16

    ppo_agent = PPOAgent(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        policy_loss_coef=args.policy_loss_coef,
        value_loss_coef=args.value_loss_coef,
        entropy_coef_initial=args.entropy_coef_initial,
        entropy_decay=args.entropy_decay,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        accumulate_episodes=args.accumulate_episodes,
        state_gnn_lr_multiplier=args.state_gnn_lr_multiplier,
            gae_lambda=args.gae_lambda,
            value_clip_epsilon=args.value_clip_epsilon,
            minibatch_size=args.minibatch_size,
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
        device=device,
        failure_penalty_ratio=args.failure_penalty_ratio
    )

    # Record
    rec = record.Record()

    # Save args
    with open(args.ckpt_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    # === DIAGNOSTIC: Track initial parameters ===
    initial_state_gnn_param = list(ppo_agent.state_gnn.parameters())[0].clone().detach()
    initial_policy_param = list(ppo_agent.policy_net.parameters())[0].clone().detach()
    initial_value_param = list(ppo_agent.value_net.parameters())[0].clone().detach()
    logger.critical("[DIAGNOSTIC] Initial parameter norms - "
                   f"StateGNN: {torch.norm(initial_state_gnn_param).item():.6f}, "
                   f"Policy: {torch.norm(initial_policy_param).item():.6f}, "
                   f"Value: {torch.norm(initial_value_param).item():.6f}")

    # Training loop
    logger.critical("Start training PPO...")
    for episode in range(args.num_epoch):
        # Training episode
        score, loss_dict = train_episode(ppo_agent, env, rec, logger)

        # Record training results
        rec.training_record["score"].append(score)
        if loss_dict:
            rec.learn_losses.append(loss_dict.get('total_loss', 0))

            # Record loss and gradient for plotting
            rec.loss_record['episodes'].append(episode + 1)
            rec.loss_record['policy_loss'].append(loss_dict.get('policy_loss', 0))
            rec.loss_record['value_loss'].append(loss_dict.get('value_loss', 0))
            rec.loss_record['entropy_loss'].append(loss_dict.get('entropy', 0))
            rec.loss_record['total_loss'].append(loss_dict.get('total_loss', 0))

            rec.gradient_record['episodes'].append(episode + 1)
            rec.gradient_record['state_gnn_grad'].append(loss_dict.get('state_gnn_grad_norm', 0))
            rec.gradient_record['policy_grad'].append(loss_dict.get('policy_grad_norm', 0))
            rec.gradient_record['value_grad'].append(loss_dict.get('value_grad_norm', 0))

            # Plot loss and gradient curves (overwrite each time)
            plot_loss_curves(rec, args.ckpt_dir)
            plot_gradient_norms(rec, args.ckpt_dir)

        logger.info(f"Episode {episode+1}/{args.num_epoch}, Score: {score:.4f}, "
                   f"Entropy Coef: {ppo_agent.get_entropy_coef():.6f}")
        if loss_dict:
            logger.info(f"  Policy Loss: {loss_dict.get('policy_loss', 0):.4f}, "
                       f"Value Loss: {loss_dict.get('value_loss', 0):.4f}, "
                       f"Entropy: {loss_dict.get('entropy', 0):.4f}")

        # Testing
        if (episode + 1) % args.test_frequency == 0:
            # === DIAGNOSTIC: Check parameter changes from initial ===
            current_state_gnn_param = list(ppo_agent.state_gnn.parameters())[0].clone().detach()
            current_policy_param = list(ppo_agent.policy_net.parameters())[0].clone().detach()
            current_value_param = list(ppo_agent.value_net.parameters())[0].clone().detach()

            state_gnn_total_change = torch.norm(current_state_gnn_param - initial_state_gnn_param).item()
            policy_total_change = torch.norm(current_policy_param - initial_policy_param).item()
            value_total_change = torch.norm(current_value_param - initial_value_param).item()

            logger.critical(f"[DIAGNOSTIC] Total parameter changes from initial (Episode {episode+1}): "
                          f"StateGNN: {state_gnn_total_change:.6f}, "
                          f"Policy: {policy_total_change:.6f}, "
                          f"Value: {value_total_change:.6f}")

            logger.critical(f"Testing at episode {episode+1}...")
            test_scores = []
            test_designs = []

            # === DIAGNOSTIC: Only log first test run in detail ===
            for run in range(args.test_runs):
                if run == 0:
                    logger.critical(f"  [DIAGNOSTIC] Detailed output for test run 1/{args.test_runs}:")
                test_score, design_process = test_episode(ppo_agent, env, rec, logger if run == 0 else None)
                test_scores.append(test_score)
                test_designs.append(design_process)

            # Compute statistics
            mean_score = np.mean(test_scores)
            std_score = np.std(test_scores)

            # === DIAGNOSTIC: Log all test scores ===
            logger.critical(f"[DIAGNOSTIC] All test scores: {[f'{s:.4f}' for s in test_scores]}")
            logger.critical(f"[DIAGNOSTIC] Test score variance: {np.var(test_scores):.6f}")

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
            plot.plot_test_behaviors(rec, env, args.ckpt_dir, args.test_frequency)

            # Save model
            save_model(ppo_agent, args.ckpt_dir, episode)

    logger.critical("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
