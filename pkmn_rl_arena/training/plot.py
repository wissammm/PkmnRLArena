from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class TrainingPlotter:
    """Visualize and save training metrics for self-play reinforcement learning"""
    
    def __init__(self, save_dir='training_plots', figsize=(18, 10), default_colors=None):
        """
        Initialize the training plotter
        
        Args:
            save_dir: Directory to save plot images
            figsize: Size of the figure (width, height)
            default_colors: Dict of plot types to colors
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        
        # Default color scheme
        self.colors = default_colors or {
            'reward': '#2E86AB',
            'win_rate': '#06A77D',
            'loss': '#D62246',
            'episode_length': '#F77F00',
            'elo': ['#2E86AB', '#F77F00', '#06A77D', '#D62246', '#5F4BB6'],
            'baseline': 'red'
        }
    
    def plot_rewards(self, ax, rewards, interval=100):
        """Plot episode rewards over time"""
        if not rewards:
            return
            
        ax.plot(rewards, linewidth=2, color=self.colors['reward'])
        ax.set_title('Average Episode Reward', fontweight='bold')
        ax.set_xlabel(f'Episode (x{interval})')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_win_rate(self, ax, win_rates, interval=100):
        """Plot win rate over time"""
        if not win_rates:
            return
            
        ax.plot(win_rates, linewidth=2, color=self.colors['win_rate'])
        ax.axhline(y=0.5, color=self.colors['baseline'], linestyle='--', alpha=0.5, label='50% baseline')
        ax.set_title('Win Rate', fontweight='bold')
        ax.set_xlabel(f'Episode (x{interval})')
        ax.set_ylabel('Win Rate')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_loss(self, ax, losses, interval=100):
        """Plot training loss over time"""
        if not losses:
            return
            
        ax.plot(losses, linewidth=2, color=self.colors['loss'])
        ax.set_title('Training Loss', fontweight='bold')
        ax.set_xlabel(f'Updates (x{interval})')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_episode_length(self, ax, lengths, interval=100):
        """Plot average episode length over time"""
        if not lengths:
            return
            
        ax.plot(lengths, linewidth=2, color=self.colors['episode_length'])
        ax.set_title('Average Episode Length', fontweight='bold')
        ax.set_xlabel(f'Episode (x{interval})')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_elo_history(self, ax, elo_history):
        """Plot ELO rating evolution over time"""
        if not elo_history or not elo_history.get('agent', []):
            return
            
        for i, (player, history) in enumerate(elo_history.items()):
            color_idx = i % len(self.colors['elo'])
            ax.plot(history, linewidth=2, label=player.capitalize(), 
                    color=self.colors['elo'][color_idx])
                    
        ax.set_title('ELO Rating Evolution', fontweight='bold')
        ax.set_xlabel('Matches')
        ax.set_ylabel('ELO Rating')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_current_elo(self, ax, current_elos, initial_elo=1500):
        """Plot current ELO ratings as a bar chart"""
        if not current_elos:
            return
            
        players = list(current_elos.keys())
        ratings = list(current_elos.values())
        
        bars = ax.bar(players, ratings, color=self.colors['elo'][:len(players)])
        ax.set_title('Current ELO Ratings', fontweight='bold')
        ax.set_ylabel('ELO Rating')
        ax.axhline(y=initial_elo, color='gray', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        return ax
    
    
    def save_training_plots(self, stats_tracker, elo_system, episode_num, win_matrix=None, 
                           action_counts=None, action_labels=None, interval=100, initial_elo=1500):
        """Generate and save comprehensive training visualization plots"""
        plot_data = stats_tracker.get_plot_data()
        elo_history = elo_system.get_rating_history() if elo_system else {}
        current_elos = elo_system.get_all_ratings() if elo_system else {}
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle(f'Training Progress - Episode {episode_num}', fontsize=16, fontweight='bold')
        
        self.plot_rewards(axes[0, 0], plot_data.get('rewards', []), interval)
        self.plot_win_rate(axes[0, 1], plot_data.get('winrates', []), interval)
        self.plot_loss(axes[0, 2], plot_data.get('losses', []), interval)
        self.plot_episode_length(axes[1, 0], plot_data.get('lengths', []), interval)
        self.plot_elo_history(axes[1, 1], elo_history)
        self.plot_current_elo(axes[1, 2], current_elos, initial_elo)
        
        plt.tight_layout()
        filepath = self.save_dir / f"training_ep_{episode_num}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        
    def save_selfplay_analysis(self, win_matrix, action_counts=None, 
                              agent_labels=None, action_labels=None, episode_num=None):
        """Generate and save self-play specific analysis plots"""
        # self.plot_win_matrix
        # self.plot_action_distribution
        