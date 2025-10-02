"""
Training functionality for Option-Critic algorithm.

This module contains the OptionCriticTrainer class that handles
the main training loop and all related functionality.
"""

import datetime
import logging
import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from RL.environment import Environment
from RL.option_critic_gnn import OptionCriticGNN, critic_loss, actor_loss
from RL.experience_replay import ReplayBuffer
from RL.record import Record
from Validation import normalization as nda_norm
from Visualization.plot import plot_test_behaviors

from .utils import get_graph_data, num_actions
from .rollout import rollout_option
from .evaluation import evaluate_model
from .logger import termination_logger


class OptionCriticTrainer:
    """Main trainer class for Option-Critic algorithm."""

    def __init__(self, args):
        """Initialize trainer with configuration arguments."""
        self.args = args
        self.device = torch.device(args.device)

        # Create timestamped checkpoint directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ckpt_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "option_oc" / timestamp
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created timestamped checkpoint directory: {self.ckpt_dir}")

        # Setup logging
        self.logger = self._setup_logger()

        # Setup termination logger with correct path (do this early)
        termination_logger.save_path = str(self.ckpt_dir / "termination_stats.json")

        # Initialize environment and models
        self.base_env = self._setup_environment()
        self.oc, self.oc_prime = self._setup_models()
        self.actor_optimizer, self.critic_optimizer = self._setup_optimizers()

        # Initialize buffer and statistics
        self.buffer = ReplayBuffer(capacity=100000)
        self.all_stats = self._initialize_stats()

        # Best model tracking
        self.best_model_score = float('-inf')
        self.best_model_path = self.ckpt_dir / "best_model.pt"
        self.best_model_info_path = self.ckpt_dir / "best_model_info.json"

        # History tracking
        self.evaluation_histories = {
            "round_best_scores": [],
            "round_best_actions": [],
            "round_best_options": [],
            "round_best_option_instances": [],
            "round_best_actions_SCWB": [],
            "round_numbers": []
        }

        self.training_histories = {
            "all_scores": [],
            "all_actions": [],
            "all_options": [],
            "all_option_instances": [],
            "all_actions_SCWB": []
        }

    def _setup_logger(self):
        """Setup logging configuration."""
        logger = logging.getLogger("OC_Train")
        if not logger.handlers:
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            log_file = self.ckpt_dir / "training.log"
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.setLevel(logging.DEBUG)
            logger.info(f"Debug logging enabled - logs saved to {log_file}")
            logger.info(f"Timestamped checkpoint directory: {self.ckpt_dir}")

        return logger

    def _setup_environment(self):
        """Setup the training environment."""
        nda_simulator = None
        nda_norm_dict = nda_norm.get_normalization_dict() if hasattr(nda_norm, "get_normalization_dict") else {}

        base_env = Environment(
            structure_shape=self.args.structure_shape,
            add_structure_geometry=self.args.add_structure_geometry,
            add_response_features=self.args.add_response_features,
            reward_type=self.args.reward_type,
            scwb_driven_design=False,
            do_nonlinear_dynamic_analysis=False,
            check_acceleration=self.args.check_acceleration,
            check_displacement=self.args.check_displacement,
            nda_simulator=nda_simulator,
            nda_norm_dict=nda_norm_dict,
            DBE_ground_motion_set=[],
            MCE_ground_motion_set=[],
            checkpoint_dir=self.ckpt_dir,
            logger=self.logger,
            device=self.device,
        )

        return base_env

    def _setup_models(self):
        """Setup Option-Critic models."""
        # Reset environment to infer feature sizes
        structure = self.base_env.reset()
        graph = structure.graph
        node_feature_dim = graph.x.shape[1]
        edge_feature_dim = graph.edge_attr.shape[1]

        # Create models
        A = num_actions(structure)
        oc = OptionCriticGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=self.args.hidden_dim,
            member_state_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            num_actions=A,
            num_options=self.args.num_options,
            temperature=self.args.temperature,
            eps_start=self.args.eps_start,
            eps_min=self.args.eps_min,
            eps_decay=self.args.eps_decay,
            eps_test=self.args.eps_test,
            device=self.device,
            testing=False,
            debug_logging=True,
        )

        oc_prime = OptionCriticGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=self.args.hidden_dim,
            member_state_dim=self.args.hidden_dim,
            num_layers=self.args.num_layers,
            num_actions=A,
            num_options=self.args.num_options,
            temperature=self.args.temperature,
            eps_start=self.args.eps_start,
            eps_min=self.args.eps_min,
            eps_decay=self.args.eps_decay,
            eps_test=self.args.eps_test,
            device=self.device,
            testing=True,
            debug_logging=False,
        )

        oc_prime.load_state_dict(oc.state_dict())

        return oc, oc_prime

    def _setup_optimizers(self):
        """Setup optimizers for different model components."""
        # Separate parameters for different components
        termination_params = [p for n, p in self.oc.named_parameters()
                             if n.startswith('terminations')]
        policy_params = [self.oc.options_W, self.oc.options_b]
        critic_params = [p for n, p in self.oc.named_parameters()
                        if n.startswith('Q')]
        state_gnn_params = [p for n, p in self.oc.named_parameters()
                           if n.startswith('state_gnn')]
        feature_params = [p for n, p in self.oc.named_parameters()
                         if n.startswith('feature_processor')]

        # Use different learning rates for different components
        gnn_lr = self.args.actor_lr * 0.1
        termination_lr = self.args.actor_lr * self.args.termination_lr_ratio

        actor_optimizer = optim.Adam([
            {"params": policy_params, "lr": self.args.actor_lr},
            {"params": termination_params, "lr": termination_lr},
            {"params": state_gnn_params, "lr": gnn_lr},
            {"params": feature_params, "lr": self.args.actor_lr},
        ])

        critic_optimizer = optim.Adam([
            {"params": critic_params, "lr": self.args.critic_lr},
            {"params": state_gnn_params, "lr": gnn_lr},
            {"params": feature_params, "lr": self.args.critic_lr},
        ])

        # Log configuration
        self.logger.info(f"Learning rate configuration:")
        self.logger.info(f"  Policy parameters: {self.args.actor_lr}")
        self.logger.info(f"  Termination parameters: {termination_lr}")
        self.logger.info(f"  Critic parameters: {self.args.critic_lr}")
        self.logger.info(f"  StateGNN parameters: {gnn_lr}")

        return actor_optimizer, critic_optimizer

    def _initialize_stats(self):
        """Initialize statistics tracking."""
        return {
            "option_lengths": [],
            "termination_reasons": {"beta": 0, "max_len": 0, "minimum_section": 0},
            "entropy_mean": [],
            "entropy_last": [],
            "epsilon": [],
            "episode_rewards": [],
            "episode_option_lengths_mean": [],
            "episode_termination_counts": [],
            "episode_entropy_mean": [],
            "episode_score": [],
            "last_pass_sections": [],
            "last_pass_saved_material": [],
            "eval_scores": [],
            "eval_success_rates": [],
            "eval_episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
        }

    def train_episode(self, episode_num):
        """Train a single episode."""
        self.logger.info(f"Starting episode {episode_num+1}/{self.args.epochs}")
        structure = self.base_env.reset(testing=False)

        done = False
        option_termination = True
        curr_option = 0
        greedy_option = 0
        steps = 0

        episode_opt_lengths = []
        episode_termination_counter = {"beta": 0, "max_len": 0, "minimum_section": 0}
        episode_entropies = []
        episode_score = 0.0
        episode_total_reward = 0.0
        last_pass_sections = None
        last_pass_saved_material = None

        # Training episode behavior tracking
        episode_action_sequence = []
        episode_option_sequence = []
        episode_option_instances = []
        current_option_instance_id = 0

        loop_iteration = 0
        max_loop_iterations = 1000

        while not done:
            loop_iteration += 1

            if loop_iteration > max_loop_iterations:
                self.logger.error(f"Episode {episode_num+1} exceeded maximum loop iterations")
                done = True
                break

            epsilon = self.oc.epsilon

            if option_termination:
                curr_option = np.random.choice(self.args.num_options) if np.random.rand() < epsilon else greedy_option

            structure, next_state, option_done, episode_done, o_stats, step_transitions, termination_reason = rollout_option(
                structure, self.base_env, self.device, self.args.max_option_len,
                curr_option, self.oc, epsilon, self.logger, self.args.option_length_bonus
            )

            # Collect training behavior data
            for transition in step_transitions:
                episode_action_sequence.append(transition["action"])
                episode_option_sequence.append(curr_option)
                episode_option_instances.append({
                    "option_index": curr_option,
                    "instance_id": current_option_instance_id,
                    "action_step": len(episode_action_sequence) - 1
                })

            if option_done:
                current_option_instance_id += 1

            # Update statistics
            episode_score, episode_total_reward, last_pass_sections, last_pass_saved_material = self._update_episode_stats(
                o_stats, step_transitions, episode_opt_lengths,
                episode_termination_counter, episode_entropies,
                episode_score, episode_total_reward,
                last_pass_sections, last_pass_saved_material)

            # Push transitions to buffer
            for tr in step_transitions:
                self.buffer.push(tr["obs"], tr["option"], tr["reward"], tr["next_obs"], tr["done"])

            # Training updates
            self._perform_updates(step_transitions, steps)

            # Update state for next iteration
            if not episode_done:
                current_graph_data = get_graph_data(structure, self.device)
                _, global_state = self.oc.get_state(*current_graph_data)
                termination_probs = self.oc.get_terminations(global_state)
                option_termination, greedy_option = self.oc.predict_option_termination(global_state, curr_option)

                # Log termination prediction
                termination_logger.log_termination_prediction(
                    option=curr_option,
                    termination_probs=termination_probs,
                    termination_decision=option_termination,
                    episode=episode_num,
                    step=loop_iteration,
                    context="main_training_loop"
                )

            done = episode_done
            steps += 1

        # Store training episode data
        if len(episode_action_sequence) > 0:
            self.training_histories["all_scores"].append(episode_score)
            self.training_histories["all_actions"].append(episode_action_sequence)
            self.training_histories["all_options"].append(episode_option_sequence)
            self.training_histories["all_option_instances"].append(episode_option_instances)
            self.training_histories["all_actions_SCWB"].append([])

        # Store last pass data (like original implementation)
        self.all_stats["last_pass_sections"].append(last_pass_sections)
        self.all_stats["last_pass_saved_material"].append(float(last_pass_saved_material) if last_pass_saved_material is not None else None)


        return episode_score, episode_total_reward, episode_opt_lengths, episode_termination_counter, episode_entropies

    def _update_episode_stats(self, o_stats, step_transitions, episode_opt_lengths,
                            episode_termination_counter, episode_entropies,
                            episode_score, episode_total_reward,
                            last_pass_sections, last_pass_saved_material):
        """Update episode statistics and return updated episode_score and episode_total_reward."""
        # Update global stats
        self.all_stats["option_lengths"].append([o_stats["option_length"]])
        self.all_stats["termination_reasons"][o_stats["termination_reason"]] += 1
        self.all_stats["entropy_mean"].append(o_stats["entropy_mean"])
        self.all_stats["entropy_last"].append(o_stats["entropy_last"])
        self.all_stats["epsilon"].append(o_stats["epsilon"] if o_stats["epsilon"] is not None else float(self.oc.epsilon))

        # Update episode stats
        episode_opt_lengths.append(o_stats["option_length"])
        episode_termination_counter[o_stats["termination_reason"]] += 1
        if not np.isnan(o_stats["entropy_mean"]):
            episode_entropies.append(o_stats["entropy_mean"])

        # Update episode score
        if bool(o_stats.get("passed", False)):
            option_total_reward = sum(tr["original_reward"] for tr in step_transitions)
            episode_score += float(option_total_reward)

            option_total_real_reward = sum(tr["reward"] for tr in step_transitions)
            episode_total_reward += float(option_total_real_reward)

            last_pass_sections = o_stats.get("story_level_sections", last_pass_sections)
            last_pass_saved_material = o_stats.get("option_saved_material", last_pass_saved_material)

        # Return updated values
        return episode_score, episode_total_reward, last_pass_sections, last_pass_saved_material

    def _perform_updates(self, step_transitions, steps):
        """Perform actor and critic updates."""
        # Accumulate actor losses
        accumulated_actor_losses = []
        if len(step_transitions) > 0:
            for tr in step_transitions:
                a_loss = actor_loss(
                    tr["obs"], tr["option"], tr["logp"], tr["entropy"], tr["reward"],
                    tr["done"], tr["next_obs"], self.oc, self.oc_prime,
                    self.args.gamma, self.args.termination_reg, self.args.entropy_reg
                )
                accumulated_actor_losses.append(a_loss)

        # Synchronous updates
        should_update = len(self.buffer) > self.args.batch_size and (steps % self.args.update_frequency == 0)
        if should_update:
            # Actor update
            if len(accumulated_actor_losses) > 0:
                total_actor_loss = sum(accumulated_actor_losses) / len(accumulated_actor_losses)

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                if self.args.grad_clip is not None and self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.oc.parameters(), max_norm=self.args.grad_clip)
                self.actor_optimizer.step()

                self.all_stats["actor_losses"].append(float(total_actor_loss.item()))

            # Critic update
            data_batch = self.buffer.sample(self.args.batch_size)
            c_loss = critic_loss(self.oc, self.oc_prime, data_batch, self.args.gamma)
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            if self.args.grad_clip is not None and self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.oc.parameters(), max_norm=self.args.grad_clip)
            self.critic_optimizer.step()

            self.all_stats["critic_losses"].append(float(c_loss.item()))

            # Update target network
            if steps % self.args.freeze_interval == 0:
                self.oc_prime.load_state_dict(self.oc.state_dict())

    def evaluate_and_save(self, episode_num):
        """Evaluate model and save if best."""
        avg_score, avg_episode_length, success_rate, eval_history = evaluate_model(
            self.base_env, self.oc, self.device, self.args.eval_episodes,
            self.args.max_option_len, self.logger, seed=42
        )

        # Update evaluation statistics
        self.all_stats["eval_scores"].append(avg_score)
        self.all_stats["eval_success_rates"].append(success_rate)
        self.all_stats["eval_episode_lengths"].append(avg_episode_length)

        # Save best model
        if avg_score > self.best_model_score:
            self.best_model_score = avg_score

            torch.save({
                'state_dict': self.oc.state_dict(),
                'hyperparameters': vars(self.args),
                'episode': episode_num + 1,
                'score': avg_score,
                'success_rate': success_rate
            }, self.best_model_path)

            best_model_info = {
                'episode': episode_num + 1,
                'score': float(avg_score),
                'success_rate': float(success_rate),
                'avg_episode_length': float(avg_episode_length),
                'timestamp': datetime.datetime.now().isoformat(),
                'hyperparameters': vars(self.args)
            }

            with open(self.best_model_info_path, 'w', encoding='utf-8') as f:
                json.dump(best_model_info, f, indent=2, ensure_ascii=False)

            self.logger.info(f"New best model saved with score {avg_score:.2f}")

        # Generate test behavior visualization after evaluation
        self.plot_test_behaviors(episode_num, eval_history)

        # Generate option preview visualization for best episode
        self.generate_option_preview_visualization(episode_num, eval_history)

        return avg_score, avg_episode_length, success_rate

    def train(self):
        """Main training loop."""
        self.logger.info("Starting Option-Critic training")

        for episode_num in range(self.args.epochs):
            # Train episode
            episode_score, episode_total_reward, episode_opt_lengths, episode_termination_counter, episode_entropies = self.train_episode(episode_num)

            # Update episode-level statistics
            self.all_stats["episode_rewards"].append(episode_total_reward)
            self.all_stats["episode_option_lengths_mean"].append(np.mean(episode_opt_lengths) if episode_opt_lengths else 0.0)
            self.all_stats["episode_termination_counts"].append(episode_termination_counter)
            self.all_stats["episode_entropy_mean"].append(np.mean(episode_entropies) if episode_entropies else 0.0)
            self.all_stats["episode_score"].append(episode_score)

            # Generate training behavior visualization after each episode
            self.plot_training_behaviors(episode_num)

            # Save statistics every episode (like original implementation)
            self._save_episode_stats()

            # Evaluate periodically
            if (episode_num + 1) % self.args.eval_frequency == 0:
                avg_score, avg_episode_length, success_rate = self.evaluate_and_save(episode_num)
                self.logger.info(f"Episode {episode_num+1}: score={episode_score:.2f}, eval_score={avg_score:.2f}, success_rate={success_rate:.1f}%")
            else:
                self.logger.info(f"Episode {episode_num+1}: score={episode_score:.2f}")

            # Note: epsilon is automatically updated when accessed via self.oc.epsilon property

        # Final evaluation and save
        self.logger.info("Training completed, performing final evaluation")
        self.evaluate_and_save(self.args.epochs - 1)

        # Save final statistics
        termination_logger.save_stats()

        stats_path = self.ckpt_dir / "oc_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_stats, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Training completed. Best model score: {self.best_model_score:.2f}")
        self.logger.info(f"Results saved to: {self.ckpt_dir}")

    def plot_training_behaviors(self, episode_num):
        """Plot training behavior visualization after each episode."""
        if len(self.training_histories["all_scores"]) == 0:
            return None

        self.logger.info(f"Generating training behavior visualization after episode {episode_num+1}...")

        # Create a Record-like object for compatibility with plot_test_behaviors
        class TrainingRecord:
            def __init__(self, training_histories):
                # For training visualization, we use training data as both training and testing
                self.training_record = {"score": training_histories["all_scores"]}
                self.testing_record = {
                    "score": training_histories["all_scores"],
                    "action": training_histories["all_actions"],
                    "action_SCWB": training_histories["all_actions_SCWB"],
                    "option": training_histories["all_options"],
                    "option_instances": training_histories["all_option_instances"]
                }

        # Create record with current training data
        training_record = TrainingRecord(self.training_histories)

        # Generate the behavior visualization
        try:
            import shutil

            # Call the original plot function
            plot_test_behaviors(training_record, self.base_env, self.ckpt_dir)

            # Rename the generated file to include episode number and indicate it's training
            testing_path = self.ckpt_dir / 'testing_behaviors.png'
            training_path = self.ckpt_dir / f'training_behaviors_ep{episode_num+1}.png'

            if testing_path.exists():
                shutil.move(str(testing_path), str(training_path))
                self.logger.info(f"Training behavior plot saved to: {training_path}")
                return training_path
            else:
                self.logger.warning(f"Failed to generate training behavior plot for episode {episode_num+1}")
                return None

        except Exception as e:
            self.logger.error(f"Error generating training behavior plot for episode {episode_num+1}: {str(e)}")
            return None

    def plot_test_behaviors(self, episode_num, eval_history):
        """Plot test behavior visualization after evaluation."""
        if not eval_history or len(eval_history["scores"]) == 0:
            return None

        self.logger.info(f"Generating test behavior visualization after inference (episode {episode_num+1})...")

        # Store the best episode from current evaluation round
        best_idx = np.argmax(eval_history["scores"])
        self.evaluation_histories["round_best_scores"].append(eval_history["scores"][best_idx])
        self.evaluation_histories["round_best_actions"].append(eval_history["actions"][best_idx])
        self.evaluation_histories["round_best_options"].append(eval_history["options"][best_idx])
        self.evaluation_histories["round_best_option_instances"].append(eval_history["option_instances"][best_idx])
        self.evaluation_histories["round_best_actions_SCWB"].append(eval_history["actions_SCWB"][best_idx])
        self.evaluation_histories["round_numbers"].append(episode_num + 1)

        # Create a Record-like object for compatibility with plot_test_behaviors
        class EvaluationRecord:
            def __init__(self, round_histories):
                # Use only the best episode from each evaluation round for visualization
                self.training_record = {"score": round_histories["round_best_scores"]}
                self.testing_record = {
                    "score": round_histories["round_best_scores"],
                    "action": round_histories["round_best_actions"],
                    "action_SCWB": round_histories["round_best_actions_SCWB"],
                    "option": round_histories["round_best_options"],
                    "option_instances": round_histories["round_best_option_instances"]
                }

        # Create record with accumulated evaluation histories
        eval_record = EvaluationRecord(self.evaluation_histories)

        # Generate the behavior visualization
        try:
            import os

            # Call the original plot function
            plot_test_behaviors(eval_record, self.base_env, self.ckpt_dir)

            # Rename the generated file to include episode number and indicate it's test
            testing_path = self.ckpt_dir / 'testing_behaviors.png'
            test_inference_path = self.ckpt_dir / f'test_behaviors_inference_ep{episode_num+1}.png'

            if testing_path.exists():
                import shutil
                shutil.move(str(testing_path), str(test_inference_path))
                file_size = os.path.getsize(test_inference_path)
                self.logger.info(f"Test behavior plot saved to: {test_inference_path} (size: {file_size} bytes)")
                return test_inference_path
            else:
                self.logger.warning(f"Failed to generate test behavior plot for episode {episode_num+1}")
                return None

        except Exception as e:
            self.logger.error(f"Error generating test behavior plot for episode {episode_num+1}: {str(e)}")
            try:
                plt.close('all')
            except:
                pass
            return None

    def generate_option_preview_visualization(self, episode_num, eval_history):
        """
        Generate option preview visualization for the best episode in this evaluation round.

        Args:
            episode_num: Current training episode number
            eval_history: Evaluation history containing episode data
        """
        # Check if option preview is enabled and should run this round
        if not getattr(self.args, 'enable_option_preview', True):
            return None

        preview_frequency = getattr(self.args, 'option_preview_frequency', 10)
        if (episode_num + 1) % preview_frequency != 0:
            return None

        # Find best episode from evaluation
        if not eval_history.get('scores'):
            self.logger.warning("No scores found in eval_history")
            return None

        best_episode_idx = np.argmax(eval_history['scores'])
        best_episode_score = eval_history['scores'][best_episode_idx]

        # Prepare best episode data
        best_episode_data = {
            'actions': eval_history['actions'][best_episode_idx],
            'options': eval_history['options'][best_episode_idx] if 'options' in eval_history else [],
            'option_instances': eval_history['option_instances'][best_episode_idx] if 'option_instances' in eval_history else []
        }

        # Check if we have the required option data
        if not best_episode_data.get('option_instances'):
            self.logger.warning("No option_instances data found, skipping option preview visualization")
            return None

        episode_info = {
            'episode_number': best_episode_idx,
            'round_number': episode_num + 1,
            'score': best_episode_score,
            'avg_score': eval_history.get('avg_score', best_episode_score)
        }

        self.logger.info(f"Generating option preview for round {episode_num+1}, best episode {best_episode_idx} with score {best_episode_score:.2f}")

        # Import and call visualization function
        from Visualization.visualize import visualize_option_preview_process

        # Create a mock agent that works with the visualization
        # The visualize_option_preview_process expects a DeepQAgent but we have OptionCriticGNN
        mock_agent = type('MockAgent', (), {
            'device': self.device,
            'gnn': self.oc,  # Pass the entire OptionCriticGNN model as 'gnn'
            'online_q_network': None,  # OptionCriticGNN doesn't have this
            'target_q_network': None   # OptionCriticGNN doesn't have this
        })()

        # Call visualization function
        visualize_option_preview_process(
            agent=mock_agent,
            env=self.base_env,
            logger=self.logger,
            save_model_path=self.best_model_path,
            episode_data=best_episode_data,
            episode_info=episode_info,
            save_base_dir=self.ckpt_dir
        )

        return True

    def _save_episode_stats(self):
        """Save episode statistics to oc_stats.json (like original implementation)."""
        try:
            stats_path = self.ckpt_dir / "oc_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_stats, f, indent=2, default=str, ensure_ascii=False)

            self.logger.debug(f"Stats saved to {stats_path}")
        except Exception as e:
            self.logger.error(f"Error saving episode stats: {str(e)}")