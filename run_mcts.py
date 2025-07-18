import json
import torch
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
from copy import deepcopy

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("RL/")
sys.path.append("Structure/")
sys.path.append("Visualization")
sys.path.append("NonlinearDynamicAnalysisSimulator/")

from RL import agent, environment, record, mcts
from Visualization import plot
from NonlinearDynamicAnalysisSimulator import load_simulator


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # General
    parser.add_argument("--comment", type=str, default="MCTS run")
    parser.add_argument("--ckpt_dir", type=Path, default="./Results/AdjustedMoreSections/RandomShape/HybridMCTS_Runs")
    parser.add_argument("--suffix", type=str, default="MCTS_Test")
    parser.add_argument("--num_epoch", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument("--random_seed", type=int, default=732, help="Fixed random seed.")

    # Algorithm
    parser.add_argument("--algorithm", type=str, default="HybridMCTS", choices=["MCTS", "HybridMCTS"], help="MCTS algorithm to use.")
    parser.add_argument("--dqn_checkpoint_dir", type=Path, default='./models/DQN/20250605_RSA_model_HighestScore.pt', help="Required for HybridMCTS. Path to a pretrained DQN agent checkpoint.")

    # MCTS Hyperparameters
    parser.add_argument("--n_simulations", type=int, default=50, help="Number of simulations per MCTS search.")
    parser.add_argument("--c_puct", type=float, default=1.0, help="Exploration constant for UCT in MCTS.")
    parser.add_argument("--rollout_depth", type=int, default=5, help="For HybridMCTS, number of random steps in rollout before using DQN.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for MCTS.")

    # Environment Arguments (copied from train.py for consistency)
    parser.add_argument("--structure_shape", type=str, default="fixed", help="fixed, small_random, random")
    parser.add_argument("--add_structure_geometry", action="store_true", default=True)
    parser.add_argument("--add_response_features", action="store_true", default=True)
    parser.add_argument("--reward_type", type=str, default="material", help="material, acceleration, displacement, normalized, total, combined")
    parser.add_argument("--scwb_driven_design", action="store_true", default=False)
    
    # NDA Arguments
    parser.add_argument("--do_nonlinear_dynamic_analysis", action="store_true", default=False)
    parser.add_argument("--check_acceleration", action="store_true", default=False)
    parser.add_argument("--check_displacement", action="store_true", default=True)
    parser.add_argument("--graph_lstm_dir", type=Path, default=None)
    parser.add_argument("--ground_motion_dir", type=Path, default=None)
    parser.add_argument("--ground_motion_number", type=int, default=11)
    
    # Model arguments (for loading DQN in HybridMCTS)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)

    args = parser.parse_args()

    if args.algorithm == "HybridMCTS" and args.dqn_checkpoint_dir is None:
        parser.error("--dqn_checkpoint_dir is required when using --algorithm HybridMCTS")

    return args


def set_random_seed(SEED: int):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_loggings(ckpt_dir):
    logger = logging.getLogger(name='MCTS-RL')
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(ckpt_dir / "mcts_record.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def run_mcts_episode(mcts_agent, env, rec, logger):
    """Runs a single episode of MCTS-driven design."""
    state = env.reset()
    rec.testing_record['initial_design'].append(state.story_level_sections)
    logger.info(f"Initial Design: {state.story_level_sections}, Volume: {state.calculate_material_usage():.3f}")

    done = False
    total_reward = 0
    step_count = 0
    last_good_state = state # Initialize with the initial state
    episode_actions = []

    while not done:
        action = mcts_agent.search(state)
        episode_actions.append(action)
        
        if action is None:
            logger.warning("MCTS search returned no action. Ending episode.")
            break

        # Before stepping, the current 'state' is a good state
        last_good_state = deepcopy(state)
        last_total_reward = deepcopy(total_reward)
            
        next_state, reward, done, fail_name, fail_reason = env.step(state, action)
        total_reward += reward
        state = next_state
        step_count += 1
    
    # Record final state
    logger.info(f"Episode Finished in {step_count} steps. Total Reward: {total_reward:.3f}")
    
    # Decide which state to record as the final one
    final_design_passed = env._check_design_feasibility(state)
    final_state_to_record = state if final_design_passed else last_good_state
    
    final_volume = final_state_to_record.calculate_material_usage()
    score = total_reward if final_design_passed else last_total_reward
    
    rec.testing_record["score"].append(score)
    rec.testing_record["final_volume"].append(final_volume)
    rec.testing_record["final_design"].append(final_state_to_record.story_level_sections)
    rec.testing_record["action"].append(episode_actions)
    rec.testing_record["fail_name"].append(fail_name)
    rec.testing_record["fail_reason"].append(fail_reason)

    logger.info(f"Final Design: {final_state_to_record.story_level_sections}, Volume: {final_volume:.3f}, Score: {score:.3f}")


def main(args):
    set_random_seed(args.random_seed)

    args.ckpt_dir = args.ckpt_dir / f'{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}__{args.suffix}'
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = get_loggings(args.ckpt_dir)
    logger.critical(args.ckpt_dir)
    logger.critical(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.critical(f"Device: {device}")
    
    # Environment
    _env_kwargs = {
        "structure_shape": args.structure_shape,
        "add_structure_geometry": args.add_structure_geometry,
        "add_response_features": args.add_response_features,
        "reward_type": args.reward_type,
        "scwb_driven_design": args.scwb_driven_design,
        "do_nonlinear_dynamic_analysis": args.do_nonlinear_dynamic_analysis,
        "check_acceleration": args.check_acceleration,
        "check_displacement": args.check_displacement,
        "nda_simulator": None, # NDA simulator not integrated with MCTS yet
        "nda_norm_dict": None,
        "DBE_ground_motion_set": None,
        "MCE_ground_motion_set": None,
        "checkpoint_dir": args.ckpt_dir,
        "logger": logger,
        "device": device,
    }
    env = environment.Environment(**_env_kwargs)

    # Load pretrained DQN agent if running HybridMCTS
    dqn_for_hybrid = None
    if args.algorithm == "HybridMCTS":
        logger.info(f"Loading pretrained DQN from {args.dqn_checkpoint_dir} for HybridMCTS.")
        # We need to know the feature dimensions from the environment to load the agent
        dummy_structure = env.reset()
        node_feature_dim = dummy_structure.graph.x.shape[1]
        edge_feature_dim = dummy_structure.graph.edge_attr.shape[1]

        dqn_for_hybrid = agent.DeepQAgent(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=args.hidden_dim, 
            num_layers=args.num_layers,
            pretrained_ckpt_dir=args.dqn_checkpoint_dir,
            device=device,
            # Dummy values for params not needed for inference
            batch_size=1, lr=0, buffer_size=1, epsilon_decay_schedule=lambda n: 0,
            synchronize_steps=1, soft_update_alpha=0, gamma=0, update_frequency=1,
            add_experience_frequency=1, logger=logger
        )

    # Initialize the MCTS Agent
    mcts_agent = mcts.MCTSAgent(
        env=env,
        algorithm=args.algorithm,
        n_simulations=args.n_simulations,
        c_puct=args.c_puct,
        dqn_agent=dqn_for_hybrid,
        rollout_depth=args.rollout_depth,
        gamma=args.gamma
    )

    rec = record.Record()

    logger.critical(f"Starting MCTS run for {args.num_epoch} episodes...")
    for i in range(args.num_epoch):
        logger.info(f"--- Starting Episode {i+1}/{args.num_epoch} ---")
        run_mcts_episode(mcts_agent, env, rec, logger)

    logger.critical("MCTS run finished.")
    logger.critical(f"Minimum Material Usage: {np.min(rec.testing_record['final_volume']):.3f} m3, Story Level Sections: {rec.testing_record['final_design'][np.argmin(rec.testing_record['final_volume'])]}")
    logger.critical(f"Highest Score: {np.max(rec.testing_record['score']):.3f}, Story Level Sections: {rec.testing_record['final_design'][np.argmax(rec.testing_record['score'])]}")

    # Plotting (using a simplified reward plot for MCTS)
    plot.plot_reward([], rec.testing_record["score"], args.ckpt_dir, is_mcts=True)
    plot.plot_fail_names([], rec.testing_record["fail_name"], args.ckpt_dir, is_mcts=True)
    plot.plot_fail_reasons([], rec.testing_record["fail_reason"], args.ckpt_dir, is_mcts=True)
    plot.plot_test_behaviors(rec, env, args.ckpt_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args) 