"""
DAC Trainer Module

This module implements the main training loop for DAC with PPO optimization.
Coordinates between environment, agent, and logging for complete training pipeline.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
import json
import datetime

from .config import DACConfig
from .agent import DACAgent
from .environment import DACEnvironmentWrapper
from .logger import DACLogger
from .utils import set_seed, EarlyStopping, MovingAverage
from .evaluation import evaluate_dac_model
from ..environment import Environment
from Validation import normalization as nda_norm
from Visualization.plot import plot_test_behaviors


class DACTrainer:
    """
    Main trainer for DAC (Double Actor-Critic) with PPO optimization.

    Manages the complete training pipeline including:
    - Episode execution with dual MDP handling
    - Experience collection and storage
    - Agent training and updates
    - Evaluation and logging
    - Checkpointing and model saving
    """

    def __init__(self,
                 config: DACConfig,
                 base_environment: Environment,
                 logger: Optional[DACLogger] = None):
        """
        Initialize DAC trainer.

        Args:
            config: DAC configuration
            base_environment: Base environment instance
            logger: Logger instance (creates new if None)
        """
        self.config = config
        self.logger = logger or DACLogger(
            log_dir=config.checkpoint_dir,
            experiment_name="dac_training"
        )

        # Set random seed
        set_seed(config.seed)

        # Initialize environment wrapper
        self.env = DACEnvironmentWrapper(
            base_env=base_environment,
            response_feature=config.add_response_features,
            length_bonus_weight=config.length_bonus_weight,
            score_tolerance=config.score_tolerance,
            logger=self.logger.logger
        )

        # Initialize agent
        self.agent = DACAgent(config, self.logger.logger)

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.best_score = -float('inf')

        # Statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []  # Track episode scores
        self.episode_final_valid_scores = []  # Track final valid scores per episode
        self.option_pass_rates = []  # Track option pass rates per episode
        self.option_usage = defaultdict(int)
        self.option_lengths = defaultdict(list)

        # Early stopping
        if config.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_threshold
            )
        else:
            self.early_stopping = None

        # Moving averages for monitoring
        self.reward_ma = MovingAverage(window_size=100)

        # Behavior tracking for visualization (similar to option_critic)
        self.training_histories = {
            "all_scores": [],
            "all_actions": [],
            "all_options": [],
            "all_option_instances": [],
            "all_actions_SCWB": []
        }

        self.evaluation_histories = {
            "round_best_scores": [],
            "round_best_actions": [],
            "round_best_options": [],
            "round_best_option_instances": [],
            "round_best_actions_SCWB": [],
            "round_numbers": []
        }

        # Evaluation parameters are now in config

        self.logger.logger.info("DAC Trainer initialized")
        self.logger.log_hyperparameters(config.to_dict())
        self.logger.log_model_summary(self.agent.network)

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training statistics
        """
        self.logger.logger.info("Starting DAC training")
        start_time = time.time()

        try:
            for episode in range(self.config.max_episodes):
                self.episode = episode

                # Run episode
                episode_stats = self._run_episode()

                # Log episode metrics
                self.logger.log_episode_metrics(episode, episode_stats)

                # Update statistics
                self._update_statistics(episode_stats)

                # Generate training behavior visualization after each episode
                self.plot_training_behaviors(episode)

                # Periodic evaluation and inference (like option_critic)
                if (episode + 1) % self.config.eval_frequency == 0:
                    avg_score, avg_episode_length, success_rate = self.evaluate_and_save(episode)
                    self.logger.logger.info(f"Episode {episode+1}: "
                                          f"train_score={episode_stats['episode_reward']:.2f}, "
                                          f"eval_score={avg_score:.2f}, "
                                          f"success_rate={success_rate:.1f}%")
                else:
                    self.logger.logger.info(f"Episode {episode+1}: "
                                          f"train_score={episode_stats['episode_reward']:.2f}")

                # Original evaluation for logging
                if episode % self.config.eval_interval == 0:
                    eval_stats = self._evaluate()
                    self.logger.log_training_metrics(episode, eval_stats)

                # Save checkpoint
                if episode % self.config.save_interval == 0:
                    self._save_checkpoint(episode)

                # Check early stopping
                if self.early_stopping and episode > 100:
                    avg_reward = np.mean(self.episode_rewards[-50:])
                    if self.early_stopping(-avg_reward, self.agent.network):  # Negative for maximization
                        self.logger.logger.info(f"Early stopping triggered at episode {episode}")
                        break

                # Progress logging
                if episode % self.config.log_interval == 0:
                    self._log_progress(episode)

        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")

        except Exception as e:
            self.logger.logger.error(f"Training failed with error: {e}")
            raise e 

        finally:
            # Final save and cleanup
            self._save_final_results()
            total_time = time.time() - start_time
            self.logger.logger.info(f"Training completed in {total_time:.2f} seconds")

        return self._get_training_summary()

    def _run_episode(self) -> Dict[str, Any]:
        """
        Run a single training episode.

        Returns:
            Episode statistics
        """
        # Reset environment
        self.logger.logger.info(f"=== Starting Episode {self.episode + 1} ===")
        dual_states, info = self.env.reset()
        self.logger.logger.info(f"Episode {self.episode + 1} environment reset completed")

        episode_reward = 0.0
        episode_steps = 0
        current_option = None
        option_start_step = 0

        # Training episode behavior tracking
        episode_action_sequence = []
        episode_option_sequence = []
        episode_option_instances = []
        current_option_instance_id = 0

        # Episode loop
        for step in range(self.config.max_steps_per_episode):
            episode_steps += 1
            self.total_steps += 1

            # Select option and action with structure object for action restrictions
            # Get current structure from environment wrapper
            current_structure = self.env._get_current_structure()
            option, action, agent_info = self.agent.select_option_and_action(
                dual_states['graph_data'],
                structure_obj=current_structure
            )
            after_selection_structure = self.env._get_current_structure()
            if after_selection_structure != current_structure:
                self.logger.logger.warning(f"After selection structure is different from current structure")
                self.logger.logger.warning(f"After selection structure: {after_selection_structure}")
                self.logger.logger.warning(f"Current structure: {current_structure}")
                raise ValueError("After selection structure is different from current structure")

            # Check for option change
            option_terminated = (current_option is not None and current_option != option)
            if current_option != option:
                if current_option is not None:
                    # Log option usage
                    option_length = step - option_start_step
                    self.option_usage[current_option] += 1
                    self.option_lengths[current_option].append(option_length)
                    # Option instance ended
                    current_option_instance_id += 1

                current_option = option
                option_start_step = step

            # Execute action
            next_dual_states, base_reward, done, env_info = self.env.step(
                action, option, option_terminated
            )

            # Collect training behavior data
            episode_action_sequence.append(action)
            episode_option_sequence.append(option)
            episode_option_instances.append({
                "option_index": option,
                "instance_id": current_option_instance_id,
                "action_step": len(episode_action_sequence) - 1
            })

            # Store experience with unified reward calculation
            self.agent.store_experience(
                dual_states['graph_data'],
                option,
                action,
                base_reward,
                next_dual_states['graph_data'],
                done,
                option_terminated,
                env_info.get('fail_name'),
                {**agent_info, **env_info}
            )

            # Update for next step
            dual_states = next_dual_states
            episode_reward += base_reward

            # Agent update
            if self.agent.should_update():
                update_stats = self.agent.update()
                if update_stats:
                    self.logger.log_training_metrics(self.total_steps, update_stats)

            # Check additional termination conditions (like Option-Critic)
            if not done:
                next_structure = env_info.get('structure')
                if next_structure is not None:
                    should_terminate, termination_reason = self.agent.check_termination_conditions(next_structure)
                    if should_terminate:
                        self.logger.logger.info(f"Episode terminated due to: {termination_reason}")
                        done = True
                        option_terminated = True

            # Check termination
            if done:
                # Log termination details
                termination_info = {
                    'fail_name': env_info.get('fail_name'),
                    'fail_reason': env_info.get('fail_reason'),
                    'episode_step': episode_steps,
                    'option_length': env_info.get('option_length', 0),
                    'material_usage': env_info.get('material_usage', 0)
                }
                if env_info.get('fail_name'):
                    self.logger.logger.warning(f"Episode {self.episode + 1} failed after {episode_steps} steps: "
                                             f"{env_info.get('fail_name')} - {env_info.get('fail_reason')}")
                else:
                    self.logger.logger.info(f"Episode {self.episode + 1} completed after {episode_steps} steps")

                self.logger.logger.debug(f"Episode termination info: {termination_info}")
                break

        # Final option logging
        if current_option is not None:
            option_length = episode_steps - option_start_step
            self.option_usage[current_option] += 1
            self.option_lengths[current_option].append(option_length)

        # Store training episode data
        if len(episode_action_sequence) > 0:
            self.training_histories["all_scores"].append(episode_reward)
            self.training_histories["all_actions"].append(episode_action_sequence)
            self.training_histories["all_options"].append(episode_option_sequence)
            self.training_histories["all_option_instances"].append(episode_option_instances)
            self.training_histories["all_actions_SCWB"].append([])

        # Collect episode statistics
        env_stats = self.env.get_episode_statistics()
        option_stats = self.env.get_option_statistics()

        episode_stats = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'material_usage': env_stats.get('current_material_usage', 0),
            'current_score': env_stats.get('current_score', 0.0),
            'final_valid_score': env_stats.get('last_valid_score', 0.0),
            'option_pass_rate': env_stats.get('option_pass_rate', 0.0),
            'total_options_passed': env_stats.get('total_options_passed', 0),
            'final_option': current_option,
            'unique_options_used': len(set(self.option_usage.keys())),
            'avg_option_length': np.mean([length for lengths in self.option_lengths.values() for length in lengths]) if self.option_lengths else 0
        }

        return episode_stats

    def _evaluate(self, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation statistics
        """
        self.logger.logger.info(f"Running evaluation with {num_episodes} episodes")

        eval_rewards = []
        eval_episode_lengths = []

        for _ in range(num_episodes):
            # Reset environment for evaluation
            dual_states, info = self.env.reset(testing=True)

            episode_reward = 0.0
            episode_steps = 0

            # Evaluation episode (deterministic policy)
            for step in range(self.config.max_steps_per_episode):
                episode_steps += 1

                # Select option and action deterministically with structure object
                current_structure = self.env._get_current_structure()
                option, action, _ = self.agent.select_option_and_action(
                    dual_states['graph_data'],
                    structure_obj=current_structure,
                    deterministic=True
                )

                # Execute action
                next_dual_states, base_reward, done, _ = self.env.step(
                    action, option, False
                )

                dual_states = next_dual_states
                episode_reward += base_reward

                if done:
                    break

            # Collect statistics
            env_stats = self.env.get_episode_statistics()
            eval_rewards.append(episode_reward)
            eval_episode_lengths.append(episode_steps)

        # Compute evaluation metrics
        eval_stats = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_episode_length_mean': np.mean(eval_episode_lengths),
            'eval_episode_length_std': np.std(eval_episode_lengths)
        }

        # Update best scores
        if eval_stats['eval_reward_mean'] > self.best_reward:
            self.best_reward = eval_stats['eval_reward_mean']
            self._save_best_model("best_reward")


        return eval_stats

    def _update_statistics(self, episode_stats: Dict[str, Any]):
        """Update training statistics."""
        self.episode_rewards.append(episode_stats['episode_reward'])
        self.episode_lengths.append(episode_stats['episode_steps'])

        # Update score statistics
        self.episode_scores.append(episode_stats.get('current_score', 0.0))
        self.episode_final_valid_scores.append(episode_stats.get('final_valid_score', 0.0))
        self.option_pass_rates.append(episode_stats.get('option_pass_rate', 0.0))

        # Update moving averages
        self.reward_ma.update(episode_stats['episode_reward'])

        # Update best score tracking
        if episode_stats.get('final_valid_score', 0.0) > self.best_score:
            self.best_score = episode_stats.get('final_valid_score', 0.0)
            self._save_best_model("best_score")

        # Log option statistics
        if self.episode % 50 == 0:
            option_stats = {
                'option_usage': dict(self.option_usage),
                'avg_option_lengths': {k: np.mean(v) for k, v in self.option_lengths.items()}
            }
            self.logger.log_option_metrics(self.episode, option_stats)

    def _log_progress(self, episode: int):
        """Log training progress."""
        recent_rewards = self.episode_rewards[-self.config.log_interval:]
        recent_scores = self.episode_final_valid_scores[-self.config.log_interval:]
        recent_pass_rates = self.option_pass_rates[-self.config.log_interval:]

        self.logger.logger.info(
            f"Episode {episode:4d} | "
            f"Avg Reward: {np.mean(recent_rewards):8.2f} | "
            f"Avg Score: {np.mean(recent_scores):6.2f}% | "
            f"Avg Pass Rate: {np.mean(recent_pass_rates)*100:5.1f}% | "
            f"Best Score: {self.best_score:6.2f}% | "
            f"Total Steps: {self.total_steps}"
        )

        # Plot option usage if available
        if episode % 100 == 0 and self.option_usage:
            self.logger.plot_option_usage(dict(self.option_usage))

    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_episode_{episode}.pt"

        # Save agent checkpoint
        self.agent.save_checkpoint(str(checkpoint_path))

        # Save training state
        training_state = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'best_score': self.best_score,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_scores': self.episode_scores,
            'episode_final_valid_scores': self.episode_final_valid_scores,
            'option_pass_rates': self.option_pass_rates,
            'option_usage': dict(self.option_usage),
            'option_lengths': {k: list(v) for k, v in self.option_lengths.items()}
        }

        training_state_path = self.config.checkpoint_dir / f"training_state_episode_{episode}.json"
        with open(training_state_path, 'w') as f:
            json.dump(training_state, f, indent=2)

        self.logger.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self, model_type: str):
        """Save best performing model (similar to option_critic approach)."""
        # Save directly in checkpoint directory like option_critic
        model_path = self.config.checkpoint_dir / "best_model.pt"
        model_info_path = self.config.checkpoint_dir / "best_model_info.json"

        # Save model checkpoint
        self.agent.save_checkpoint(str(model_path))

        # Save model info like option_critic
        best_model_info = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'score': float(self.best_reward),
            'best_score': float(self.best_score),
            'model_type': model_type,
            'hyperparameters': self.config.to_dict()
        }

        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)

        self.logger.logger.info(f"New best model saved with score {self.best_reward:.2f} (type: {model_type})")

    def _save_final_results(self):
        """Save final training results (similar to option_critic approach)."""
        # Save final checkpoint
        final_checkpoint = self.config.checkpoint_dir / "final_checkpoint.pt"
        self.agent.save_checkpoint(str(final_checkpoint))

        # Save training curves in checkpoint directory
        metrics_to_plot = ['episode_reward', 'episode_steps']
        available_metrics = {k: getattr(self, f"{k}s", []) for k in ['episode_reward']
                           if hasattr(self, f"{k}s")}
        available_metrics['episode_steps'] = self.episode_lengths

        self.logger.plot_training_curves(
            list(available_metrics.keys()),
            self.config.checkpoint_dir / "final_training_curves.png"
        )

        # Save final statistics in oc_stats.json format like option_critic
        final_stats = self._get_training_summary()
        stats_path = self.config.checkpoint_dir / "dac_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, default=str, ensure_ascii=False)

        self.logger.logger.info(f"Training completed. Best model score: {self.best_reward:.2f}")
        self.logger.logger.info(f"Results saved to: {self.config.checkpoint_dir}")

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'total_episodes': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'best_score': self.best_score,
            'final_reward_avg': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'final_score_avg': np.mean(self.episode_final_valid_scores[-100:]) if self.episode_final_valid_scores else 0,
            'final_pass_rate_avg': np.mean(self.option_pass_rates[-100:]) if self.option_pass_rates else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'option_usage_counts': dict(self.option_usage),
            'avg_option_lengths': {k: np.mean(v) for k, v in self.option_lengths.items()},
            'agent_statistics': self.agent.get_statistics()
        }

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load agent checkpoint
        self.agent.load_checkpoint(checkpoint_path)

        # Try to load training state
        training_state_path = Path(checkpoint_path).parent / f"training_state_{Path(checkpoint_path).stem}.json"
        if training_state_path.exists():
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)

            self.episode = training_state['episode']
            self.total_steps = training_state['total_steps']
            self.best_reward = training_state['best_reward']
            self.best_score = training_state.get('best_score', -float('inf'))
            self.episode_rewards = training_state['episode_rewards']
            self.episode_lengths = training_state['episode_lengths']
            self.episode_scores = training_state.get('episode_scores', [])
            self.episode_final_valid_scores = training_state.get('episode_final_valid_scores', [])
            self.option_pass_rates = training_state.get('option_pass_rates', [])
            self.option_usage = defaultdict(int, training_state['option_usage'])
            self.option_lengths = defaultdict(list,
                                            {k: list(v) for k, v in training_state['option_lengths'].items()})

            self.logger.logger.info(f"Training state loaded from {training_state_path}")

    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Training statistics
        """
        self.load_checkpoint(checkpoint_path)
        self.logger.logger.info(f"Resuming training from episode {self.episode}")

        # Continue training
        return self.train()

    def evaluate_and_save(self, episode_num):
        """Evaluate model and save if best (similar to option_critic)."""
        avg_score, avg_episode_length, success_rate, eval_history = evaluate_dac_model(
            self.agent, self.env, self.config.eval_episodes, self.logger.logger, seed=42
        )

        # Update best scores (using existing logic)
        if avg_score > self.best_reward:
            self.best_reward = avg_score
            self._save_best_model("best_reward")

        # Generate test behavior visualization after evaluation
        self.plot_test_behaviors(episode_num, eval_history)

        return avg_score, avg_episode_length, success_rate

    def plot_training_behaviors(self, episode_num):
        """Plot training behavior visualization after each episode (similar to option_critic)."""
        if len(self.training_histories["all_scores"]) == 0:
            return None

        self.logger.logger.info(f"Generating training behavior visualization after episode {episode_num+1}...")

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
        import shutil

        # Get the actual base environment from the wrapper
        base_env = getattr(self.env, 'base_env', self.env)
        # Call the original plot function (save to checkpoint directory like option_critic)
        plot_test_behaviors(training_record, base_env, self.config.checkpoint_dir)

        # Rename the generated file to include episode number and indicate it's training
        testing_path = self.config.checkpoint_dir / 'testing_behaviors.png'
        training_path = self.config.checkpoint_dir / f'training_behaviors_ep{episode_num+1}.png'

        if testing_path.exists():
            shutil.move(str(testing_path), str(training_path))
            self.logger.logger.info(f"Training behavior plot saved to: {training_path}")
            return training_path
        else:
            self.logger.logger.warning(f"Failed to generate training behavior plot for episode {episode_num+1}")
            return None



    def plot_test_behaviors(self, episode_num, eval_history):
        """Plot test behavior visualization after evaluation (similar to option_critic)."""
        if not eval_history or len(eval_history["scores"]) == 0:
            return None

        self.logger.logger.info(f"Generating test behavior visualization after inference (episode {episode_num+1})...")

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
        
        import os

        # Get the actual base environment from the wrapper
        base_env = getattr(self.env, 'base_env', self.env)
        # Call the original plot function (save to checkpoint directory like option_critic)
        plot_test_behaviors(eval_record, base_env, self.config.checkpoint_dir)

        # Rename the generated file to include episode number and indicate it's test
        testing_path = self.config.checkpoint_dir / 'testing_behaviors.png'
        test_inference_path = self.config.checkpoint_dir / f'test_behaviors_inference_ep{episode_num+1}.png'

        if testing_path.exists():
            import shutil
            shutil.move(str(testing_path), str(test_inference_path))
            file_size = os.path.getsize(test_inference_path)
            self.logger.logger.info(f"Test behavior plot saved to: {test_inference_path} (size: {file_size} bytes)")
            return test_inference_path
        else:
            self.logger.logger.warning(f"Failed to generate test behavior plot for episode {episode_num+1}")
            return None

        