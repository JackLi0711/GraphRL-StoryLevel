"""
Logging utilities for Option-Critic algorithm.

This module contains the TerminationProbabilityLogger class for tracking
and analyzing termination probability predictions during training.
"""

import datetime
import json
import numpy as np
import torch


class TerminationProbabilityLogger:
    """
    Logger for Option-Critic termination probabilities.
    Collects and saves termination statistics to JSON files.
    """

    def __init__(self, save_path="oc_stats.json"):
        self.save_path = save_path
        self.stats = {
            "termination_probabilities": [],
            "episode_stats": [],
            "global_stats": {
                "total_terminations": 0,
                "total_predictions": 0,
                "avg_termination_prob": 0.0
            }
        }
        self.episode_terminations = []
        self.current_episode = 0

    def log_termination_prediction(self, option, termination_probs, termination_decision, episode=None, step=None, context=""):
        """
        Log a termination probability prediction.

        Args:
            option: current option index
            termination_probs: tensor of termination probabilities for all options
            termination_decision: boolean decision for current option
            episode: episode number (optional)
            step: step number (optional)
            context: additional context string
        """
        # Convert tensor to list if needed
        if isinstance(termination_probs, torch.Tensor):
            if termination_probs.dim() > 1:
                # Handle multi-dimensional tensor (e.g., [batch_size, num_options])
                # Take the mean across batch dimension or first sample
                if termination_probs.shape[0] > 1:
                    termination_probs = termination_probs.mean(dim=0)  # Average across batch
                else:
                    termination_probs = termination_probs[0]  # Take first sample
            termination_probs = termination_probs.detach().cpu().numpy().tolist()

        entry = {
            "episode": episode if episode is not None else self.current_episode,
            "step": step,
            "option": int(option),
            "termination_probs": termination_probs,
            "termination_decision": bool(termination_decision),
            "context": context,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.stats["termination_probabilities"].append(entry)

        # Track for episode stats
        if termination_decision:
            # termination_probs should now be a list after conversion above
            if isinstance(termination_probs, (list, np.ndarray)) and len(termination_probs) > option:
                term_prob = float(termination_probs[option])
            else:
                term_prob = 0.0
                print(f"DEBUG: Cannot extract termination prob for option {option} from {termination_probs}")

            self.episode_terminations.append({
                "option": int(option),
                "termination_prob": term_prob,
                "step": step
            })

        # Update global stats
        self.stats["global_stats"]["total_predictions"] += 1
        if termination_decision:
            self.stats["global_stats"]["total_terminations"] += 1

        # Update average termination probability
        # termination_probs should now be a list after conversion above
        if isinstance(termination_probs, (list, np.ndarray)) and len(termination_probs) > option:
            current_prob = float(termination_probs[option])
        else:
            current_prob = 0.0
            print(f"DEBUG: Cannot extract termination prob for option {option} from {termination_probs}")

        total_preds = self.stats["global_stats"]["total_predictions"]
        prev_avg = self.stats["global_stats"]["avg_termination_prob"]
        self.stats["global_stats"]["avg_termination_prob"] = (prev_avg * (total_preds - 1) + current_prob) / total_preds

    def end_episode(self):
        """Mark the end of an episode and save episode statistics."""
        episode_stats = {
            "episode": self.current_episode,
            "num_terminations": len(self.episode_terminations),
            "terminations": self.episode_terminations.copy(),
            "avg_termination_prob": np.mean([t["termination_prob"] for t in self.episode_terminations]) if self.episode_terminations else 0.0
        }

        self.stats["episode_stats"].append(episode_stats)
        self.episode_terminations.clear()
        self.current_episode += 1

    def save_stats(self):
        """Save statistics to JSON file."""
        with open(self.save_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"Termination probability stats saved to {self.save_path}")

    def print_summary(self):
        """Print a summary of collected statistics."""
        global_stats = self.stats["global_stats"]
        print(f"\n=== Termination Probability Summary ===")
        print(f"Total predictions: {global_stats['total_predictions']}")
        print(f"Total terminations: {global_stats['total_terminations']}")
        print(f"Termination rate: {global_stats['total_terminations'] / max(1, global_stats['total_predictions']) * 100:.2f}%")
        print(f"Average termination probability: {global_stats['avg_termination_prob']:.4f}")
        print(f"Episodes recorded: {len(self.stats['episode_stats'])}")


# Global logger instance
termination_logger = TerminationProbabilityLogger()