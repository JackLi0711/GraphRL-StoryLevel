"""
DAC Logger Module

This module provides logging and visualization utilities for DAC training.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from collections import defaultdict, deque
import time

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DACLogger:
    """
    Logger for DAC training with support for:
    - Console logging
    - File logging
    - TensorBoard logging
    - Training metrics tracking
    - Visualization
    """

    def __init__(self,
                 log_dir: Path,
                 experiment_name: str = "dac_experiment",
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 use_tensorboard: bool = True):
        """
        Initialize DAC logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            console_level: Console logging level
            file_level: File logging level
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.logger = self._setup_logger(console_level, file_level)

        # Initialize TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        else:
            self.tb_writer = None
            if use_tensorboard:
                self.logger.warning("TensorBoard not available, disabling TensorBoard logging")

        # Metrics tracking
        self.metrics = defaultdict(list)
        self.scalars = defaultdict(list)
        self.step_count = 0

        # Performance tracking
        self.start_time = time.time()
        self.episode_times = deque(maxlen=100)
        self.last_episode_time = time.time()

        self.logger.info(f"DAC Logger initialized for experiment: {experiment_name}")

    def _setup_logger(self, console_level: int, file_level: int) -> logging.Logger:
        """Setup console and file logging."""
        logger = logging.getLogger(f"dac_{self.experiment_name}")
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        logger.handlers.clear()

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None,
                  log_level: int = 1):
        """
        Log scalar value.

        Args:
            tag: Metric name
            value: Scalar value
            step: Step number (uses internal counter if None)
            log_level: Logging verbosity level (higher = more verbose)
        """
        if step is None:
            step = self.step_count

        # Store metric
        self.scalars[tag].append(value)

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)

        # Console logging for important metrics
        if log_level <= 1:
            self.logger.info(f"{tag}: {value:.4f}")

    def log_episode_metrics(self, episode: int, metrics: Dict[str, Any]):
        """
        Log episode-level metrics.

        Args:
            episode: Episode number
            metrics: Dictionary of metrics
        """
        # Update episode timing
        current_time = time.time()
        episode_time = current_time - self.last_episode_time
        self.episode_times.append(episode_time)
        self.last_episode_time = current_time

        # Log basic episode info
        self.logger.info(f"Episode {episode:4d} completed in {episode_time:.2f}s")

        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"episode/{key}", value, episode)
                self.metrics[key].append(value)

        # Log performance metrics
        if len(self.episode_times) > 1:
            avg_episode_time = np.mean(self.episode_times)
            self.log_scalar("performance/avg_episode_time", avg_episode_time, episode)

        # Detailed logging
        reward = metrics.get('episode_reward', 0)
        steps = metrics.get('episode_steps', 0)
        material_saved = metrics.get('material_saved', 0)

        self.logger.info(
            f"  Reward: {reward:8.2f} | Steps: {steps:3d} | "
            f"Material Saved: {material_saved:6.2f}"
        )

    def log_training_metrics(self, step: int, metrics: Dict[str, Any]):
        """
        Log training-level metrics.

        Args:
            step: Training step
            metrics: Dictionary of training metrics
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"training/{key}", value, step, log_level=3)

    def log_option_metrics(self, step: int, option_stats: Dict[str, Any]):
        """
        Log option-specific metrics.

        Args:
            step: Step number
            option_stats: Option statistics
        """
        for key, value in option_stats.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"options/{key}", value, step, log_level=2)
            elif isinstance(value, (list, np.ndarray)):
                # Log distribution statistics
                if len(value) > 0:
                    self.log_scalar(f"options/{key}_mean", np.mean(value), step, log_level=3)
                    self.log_scalar(f"options/{key}_std", np.std(value), step, log_level=3)

    def log_network_metrics(self, step: int, network_stats: Dict[str, Any]):
        """
        Log network-specific metrics.

        Args:
            step: Step number
            network_stats: Network statistics
        """
        for key, value in network_stats.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"network/{key}", value, step, log_level=4)

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hparams: Hyperparameter dictionary
        """
        if self.tb_writer:
            # Filter out non-scalar values for TensorBoard
            scalar_hparams = {k: v for k, v in hparams.items()
                            if isinstance(v, (int, float, str, bool))}
            self.tb_writer.add_hparams(scalar_hparams, {})

        # Save to file
        hparams_file = self.log_dir / "hyperparameters.json"
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)

        self.logger.info(f"Hyperparameters saved to {hparams_file}")

    def log_model_summary(self, model, input_shape: Optional[tuple] = None):
        """
        Log model summary.

        Args:
            model: PyTorch model
            input_shape: Input shape for summary
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model Summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_text("model/total_params", str(total_params))
            self.tb_writer.add_text("model/trainable_params", str(trainable_params))

    def save_metrics(self, filename: str = "metrics.json"):
        """Save all metrics to file."""
        metrics_file = self.log_dir / filename

        # Convert deques and numpy arrays to lists
        serializable_metrics = {}
        for key, values in self.metrics.items():
            if isinstance(values, deque):
                serializable_metrics[key] = list(values)
            elif isinstance(values, np.ndarray):
                serializable_metrics[key] = values.tolist()
            else:
                serializable_metrics[key] = values

        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        self.logger.info(f"Metrics saved to {metrics_file}")

    def plot_training_curves(self, metrics_to_plot: Optional[List[str]] = None,
                           save_path: Optional[Path] = None):
        """
        Plot training curves.

        Args:
            metrics_to_plot: List of metrics to plot (plots all if None)
            save_path: Path to save plot (saves to log_dir if None)
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping plot generation")
            return

        if metrics_to_plot is None:
            metrics_to_plot = list(self.metrics.keys())

        if not metrics_to_plot:
            self.logger.warning("No metrics to plot")
            return

        # Create subplots
        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics and self.metrics[metric]:
                axes[i].plot(self.metrics[metric])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_path is None:
            save_path = self.log_dir / "training_curves.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training curves saved to {save_path}")

    def plot_option_usage(self, option_counts: Dict[int, int],
                         save_path: Optional[Path] = None):
        """
        Plot option usage distribution.

        Args:
            option_counts: Dictionary mapping option ID to usage count
            save_path: Path to save plot
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if save_path is None:
            save_path = self.log_dir / "option_usage.png"

        plt.figure(figsize=(8, 6))
        options = list(option_counts.keys())
        counts = list(option_counts.values())

        plt.bar(options, counts)
        plt.xlabel('Option ID')
        plt.ylabel('Usage Count')
        plt.title('Option Usage Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Option usage plot saved to {save_path}")

    def log_gradient_info(self, model, step: int):
        """
        Log gradient information.

        Args:
            model: PyTorch model
            step: Training step
        """
        total_norm = 0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                # Log individual parameter gradients
                self.log_scalar(f"gradients/{name}", param_norm.item(), step, log_level=5)

        total_norm = total_norm ** (1. / 2)
        self.log_scalar("gradients/total_norm", total_norm, step, log_level=3)

    def increment_step(self):
        """Increment internal step counter."""
        self.step_count += 1

    def close(self):
        """Close logger and save final metrics."""
        # Save final metrics
        self.save_metrics()

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

        # Log final summary
        total_time = time.time() - self.start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Logs saved to {self.log_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()