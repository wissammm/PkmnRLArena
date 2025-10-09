import torch
import numpy as np
from multiprocessing import Queue, Event
from random import random

from pkmn_rl_arena.env.battle_arena import BattleArena, RenderMode
from pkmn_rl_arena.env.battle_core import BattleCore
from pkmn_rl_arena.paths import PATHS

class Worker:
    def __init__(self, worker_id, model, config, experience_queue, stats_queue, epsilon_queue, stop_event, device, reset_options = []):
        self.id = worker_id
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        
        self.experience_queue = experience_queue
        self.stats_queue = stats_queue
        self.epsilon_queue = epsilon_queue
        self.stop_event = stop_event

        self.core = BattleCore(PATHS["ROM"], PATHS["BIOS"], PATHS["MAP"])
        self.env = BattleArena(self.core, render_mode=RenderMode.DISABLED)
        
        self.obs = None
        self.episode_count = 0

        self.local_steps = 0
        self.total_rewards = []
        self.wins = 0
        self.reset_options = reset_options
    
    def reset_env(self):
        self.obs, _ = self.env.reset(self.reset_options)
        return self.obs

    def sync_model(self, shared_state_dict):
        """Synchronize local model with shared parameters"""
        try:
            self.model.load_state_dict(shared_state_dict)
        except Exception as e:
            print(f"[Worker {self.id}] Sync failed: {e}")

    def get_epsilon(self, default):
        """Get current epsilon value from queue or return default"""
        try:
            return self.epsilon_queue.get_nowait()
        except:
            return default

    def select_action(self, state, epsilon):
        """Epsilon-greedy with action mask"""
        if random() < epsilon:
            valid = np.where(state["action_mask"] == 1)[0]
            return np.random.choice(valid) if len(valid) > 0 else 0
        else:
            with torch.no_grad():
                obs = torch.FloatTensor(state["observation"]).unsqueeze(0).to(self.device)
                q_values = self.model(obs).squeeze(0)
                mask = torch.FloatTensor(state["action_mask"]).to(self.device)
                q_values = q_values * mask - 1e6 * (1 - mask)
                return q_values.argmax().item()

    def run_episode(self, epsilon):
        """Epsidode """
        obs = self.reset_env()
        done = False
        episode_rewards = {"player": 0, "enemy": 0}
        steps = 0

        while not done and not self.stop_event.is_set():
            actions = {}
            for agent in self.env.agents:
                actions[agent] = self.select_action(obs[agent], epsilon)

            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
            done = all(terminations.values()) or all(truncations.values())

            for agent in self.env.agents:
                shaped_reward = rewards[agent] / 10.0
                transition = (
                    obs[agent],
                    actions[agent],
                    shaped_reward,
                    next_obs[agent],
                    done
                )
                self.experience_queue.put(transition)
                episode_rewards[agent] += shaped_reward

            obs = next_obs
            steps += 1
            self.local_steps += 1

        total_reward = sum(episode_rewards.values())
        won = episode_rewards["player"] > episode_rewards["enemy"]

        self.stats_queue.put((total_reward, steps, won, 1.0 if won else 0.0))
        self.total_rewards.append(total_reward)
        if won: self.wins += 1

    def loop(self, shared_state_dict):
        """Worker main loop"""
        while not self.stop_event.is_set():
            # sync model
            if self.episode_count % 10 == 0:
                self.sync_model(shared_state_dict)
            
            epsilon = self.get_epsilon(self.config.epsilon_end)
            self.run_episode(epsilon)
            self.episode_count += 1
    
    def play_against():
        """Play a single episode against a model for evaluation"""
        pass 
        
    