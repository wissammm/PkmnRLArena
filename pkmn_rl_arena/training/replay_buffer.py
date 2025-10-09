from collections import deque
import threading
import numpy as np
import random


class ReplayBuffer:
    """A simple replay buffer for storing transitions."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


class SharedReplayBuffer:
    """Thread-safe replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def push(self, transition):
        with self.lock:
            self.buffer.append(transition)
    
    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        with self.lock:
            return len(self.buffer)
