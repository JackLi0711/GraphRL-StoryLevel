import psutil
import torch
import gc
import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps


class MemoryMonitor:
    """Memory monitoring utility for tracking CPU and GPU memory usage."""
    
    def __init__(self, log_interval: int = 1):
        """
        Initialize memory monitor.
        
        Args:
            log_interval (int): Interval in seconds between memory logs
        """
        self.log_interval = log_interval
        self.memory_logs: List[Dict] = []
        self.start_time = None
        self.gpu_available = torch.cuda.is_available()
        
    def get_current_memory(self) -> Dict:
        """Get current memory usage statistics."""
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        memory_stats = {
            'timestamp': time.time() - (self.start_time or time.time()),
            'cpu_rss_mb': cpu_memory.rss / 1024 / 1024,  # Resident Set Size in MB
            'cpu_vms_mb': cpu_memory.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'system_available_mb': system_memory.available / 1024 / 1024,
            'system_used_percent': system_memory.percent,
        }
        
        # GPU memory if available
        if self.gpu_available:
            gpu_memory = torch.cuda.memory_stats()
            memory_stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'gpu_max_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
            })
        
        return memory_stats
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.start_time = time.time()
        self.memory_logs = []
        print(f"Memory monitoring started. GPU available: {self.gpu_available}")
        
    def log_memory(self, label: str = ""):
        """Log current memory usage with optional label."""
        stats = self.get_current_memory()
        stats['label'] = label
        self.memory_logs.append(stats)
        
        print(f"[{stats['timestamp']:.2f}s] {label} - "
              f"CPU: {stats['cpu_rss_mb']:.3f}MB, "
              f"System: {stats['system_used_percent']:.1f}%"
              + (f", GPU: {stats['gpu_allocated_mb']:.3f}MB" if self.gpu_available else ""))
    
    def cleanup_memory(self):
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        # gc.collect()
        # gc.collect()
        # gc.set_threshold(50, 5, 5)  # Default is usually (700, 10, 10); lower values increase frequency
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_peak_memory(self) -> Dict:
        """Get peak memory usage statistics."""
        if not self.memory_logs:
            return {}
            
        peak_stats = {
            'peak_cpu_rss_mb': max(log['cpu_rss_mb'] for log in self.memory_logs),
            'peak_system_used_percent': max(log['system_used_percent'] for log in self.memory_logs),
        }
        
        if self.gpu_available:
            peak_stats.update({
                'peak_gpu_allocated_mb': max(log['gpu_allocated_mb'] for log in self.memory_logs),
                'peak_gpu_reserved_mb': max(log['gpu_reserved_mb'] for log in self.memory_logs),
            })
            
        return peak_stats
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage over time."""
        if not self.memory_logs:
            print("No memory logs to plot.")
            return
            
        timestamps = [log['timestamp'] for log in self.memory_logs]
        cpu_memory = [log['cpu_rss_mb'] for log in self.memory_logs]
        system_percent = [log['system_used_percent'] for log in self.memory_logs]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU memory plot
        axes[0].plot(timestamps, cpu_memory, 'b-', label='Process RSS Memory')
        axes[0].set_ylabel('Memory (MB)')
        axes[0].set_title('Process Memory Usage Over Time')
        axes[0].legend()
        axes[0].grid(True)
        
        # System memory plot
        axes[1].plot(timestamps, system_percent, 'r-', label='System Memory %')
        axes[1].set_ylabel('Usage (%)')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_title('System Memory Usage Over Time')
        axes[1].legend()
        axes[1].grid(True)
        
        # GPU memory plot if available
        if self.gpu_available:
            gpu_allocated = [log['gpu_allocated_mb'] for log in self.memory_logs]
            gpu_reserved = [log['gpu_reserved_mb'] for log in self.memory_logs]
            
            fig.add_subplot(3, 1, 3)
            plt.plot(timestamps, gpu_allocated, 'g-', label='GPU Allocated')
            plt.plot(timestamps, gpu_reserved, 'orange', label='GPU Reserved')
            plt.ylabel('Memory (MB)')
            plt.xlabel('Time (seconds)')
            plt.title('GPU Memory Usage Over Time')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory usage plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def report_summary(self):
        """Print a summary of memory usage."""
        if not self.memory_logs:
            print("No memory logs available.")
            return
            
        peak_stats = self.get_peak_memory()
        final_stats = self.memory_logs[-1]
        
        print("\n" + "="*50)
        print("MEMORY USAGE SUMMARY")
        print("="*50)
        print(f"Total monitoring time: {final_stats['timestamp']:.1f} seconds")
        print(f"Number of memory logs: {len(self.memory_logs)}")
        print(f"\nPeak CPU Memory (RSS): {peak_stats['peak_cpu_rss_mb']:.1f} MB")
        print(f"Peak System Memory Usage: {peak_stats['peak_system_used_percent']:.1f}%")
        
        if self.gpu_available:
            print(f"Peak GPU Allocated: {peak_stats['peak_gpu_allocated_mb']:.1f} MB")
            print(f"Peak GPU Reserved: {peak_stats['peak_gpu_reserved_mb']:.1f} MB")
        
        print("="*50)


def memory_profile(monitor: MemoryMonitor, label: str = ""):
    """Decorator to profile memory usage of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor.log_memory(f"Before {label or func.__name__}")
            result = func(*args, **kwargs)
            monitor.log_memory(f"After {label or func.__name__}")
            return result
        return wrapper
    return decorator


def monitor_graph_operations(monitor: MemoryMonitor):
    """Context manager for monitoring graph operations."""
    class GraphMonitor:
        def __enter__(self):
            monitor.log_memory("Before graph operations")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            monitor.log_memory("After graph operations")
            monitor.cleanup_memory()
    
    return GraphMonitor()
