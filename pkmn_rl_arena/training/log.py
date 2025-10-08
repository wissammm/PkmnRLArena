from collections import deque
import threading
import time
import numpy as np

class StatsTracker:
    """Track and display training statistics with configurable history recording"""
    def __init__(self, window_size=100, history_interval=100):
        self.window_size = window_size
        self.history_interval = history_interval
        
        # Rolling windows for recent statistics
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.win_rates = deque(maxlen=window_size)
        self.losses = deque(maxlen=1000)
        
        # Full history for plotting
        self.full_rewards_history = []
        self.full_winrate_history = []
        self.full_loss_history = []
        self.full_length_history = []
        
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def add_episode(self, reward, length, won):
        """Record results from a completed episode"""
        with self.lock:
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.win_rates.append(1.0 if won else 0.0)
            self.total_episodes += 1
            self.total_steps += length
            
            # Update history at specified intervals
            self.update_history_if_needed()
    
    def update_history_if_needed(self):
        """Update history collections if interval is reached"""
        if self.history_interval > 0 and self.total_episodes % self.history_interval == 0:
            self.full_rewards_history.append(np.mean(self.episode_rewards))
            self.full_winrate_history.append(np.mean(self.win_rates))
            self.full_length_history.append(np.mean(self.episode_lengths))
    
    def update_history_now(self):
        """Force update of history collections regardless of interval"""
        with self.lock:
            if self.episode_rewards:  # Only update if we have data
                self.full_rewards_history.append(np.mean(self.episode_rewards))
                self.full_winrate_history.append(np.mean(self.win_rates))
                self.full_length_history.append(np.mean(self.episode_lengths))
    
    def add_loss(self, loss, update_interval=100):
        """Record training loss and update history at specified interval"""
        with self.lock:
            self.losses.append(loss)
            if len(self.losses) % update_interval == 0:
                self.full_loss_history.append(np.mean(list(self.losses)[-update_interval:]))
    
    def get_stats(self):
        """Get current training statistics"""
        with self.lock:
            if not self.episode_rewards:
                return None
            
            elapsed = time.time() - self.start_time
            return {
                'episodes': self.total_episodes,
                'steps': self.total_steps,
                'avg_reward': np.mean(self.episode_rewards),
                'avg_length': np.mean(self.episode_lengths),
                'win_rate': np.mean(self.win_rates),
                'avg_loss': np.mean(self.losses) if self.losses else 0.0,
                'eps_per_sec': self.total_episodes / elapsed if elapsed > 0 else 0,
                'steps_per_sec': self.total_steps / elapsed if elapsed > 0 else 0,
                'elapsed': elapsed
            }
    
    def get_plot_data(self):
        """Get history data suitable for plotting"""
        with self.lock:
            return {
                'rewards': list(self.full_rewards_history),
                'winrates': list(self.full_winrate_history),
                'losses': list(self.full_loss_history),
                'lengths': list(self.full_length_history)
            }
    
    def reset(self):
        """Reset all statistics"""
        with self.lock:
            self.episode_rewards.clear()
            self.episode_lengths.clear()
            self.win_rates.clear()
            self.losses.clear()
            self.full_rewards_history.clear()
            self.full_winrate_history.clear()
            self.full_loss_history.clear()
            self.full_length_history.clear()
            self.total_episodes = 0
            self.total_steps = 0
            self.start_time = time.time()

    @staticmethod
    def print_stats(stats, epsilon=None, elo_system=None, buffer_size=None):
        """Print formatted training statistics
        
        Args:
            stats: Dictionary of statistics from StatsTracker.get_stats()
            epsilon: Current exploration rate (optional)
            elo_system: ELO rating system instance (optional)
            buffer_size: Size of experience replay buffer (optional)
        """
        if stats is None:
            return
        
        print("\n" + "="*90)
        print(f"{'TRAINING STATISTICS':^90}")
        print("="*90)
        
        print(f"Episodes: {stats['episodes']:>8}  |  Total Steps: {stats['steps']:>10}")
        
        if epsilon is not None:
            print(f"Epsilon:  {epsilon:>8.4f}", end="")
            if buffer_size is not None:
                print(f"  |  Buffer Size: {buffer_size:>10}")
            else:
                print()
        
        print("-"*90)
        print(f"Avg Reward:      {stats['avg_reward']:>8.2f}  |  Win Rate:     {stats['win_rate']:>6.1%}")
        print(f"Avg Length:      {stats['avg_length']:>8.1f}  |  Avg Loss:     {stats['avg_loss']:>8.4f}")
        print(f"Episodes/sec:    {stats['eps_per_sec']:>8.2f}  |  Steps/sec:    {stats['steps_per_sec']:>8.1f}")
        
        if elo_system is not None:
            elos = elo_system.get_all_ratings()
            print("-"*90)
            print(f"ELO Ratings: Agent={elos.get('agent', 0):.0f} | Random={elos.get('random', 0):.0f} | Baseline={elos.get('baseline', 0):.0f}")
        
        print("-"*90)
        print(f"Training Time: {stats['elapsed']:.1f}s ({stats['elapsed']/60:.1f}m)")
        print("="*90)

class LogObsRewards:
    def add_observation():
        pass
