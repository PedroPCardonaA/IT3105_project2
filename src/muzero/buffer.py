from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import jax.numpy as jnp

@dataclass
class StepData:
    obs: jnp.ndarray        
    action: int            
    reward: float          
    policy: jnp.ndarray     
    root_value: float      
    done: bool

class Episode:
    def __init__(self):
        self.steps: List[StepData] = []

    def add(self, step_data: StepData) -> None:
        self.steps.append(step_data)

    def __len__(self) -> int:
        return len(self.steps)
    

class EpisodeBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: List[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample(self, batch_size: int, unroll_steps: int, td_steps: int) -> List[Tuple]:
        sampled_data = []
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            start_index = random.randint(0, len(episode) - 1)
            for unroll_step in range(unroll_steps + 1):
                index = start_index + unroll_step
                if index >= len(episode):
                    break

                obs = episode.steps[index].obs
                action = episode.steps[index].action

                target_values = []
                target_rewards = []
                for t in range(td_steps):
                    td_index = index + t
                    if td_index < len(episode):
                        target_values.append(episode.steps[td_index].root_value)
                        target_rewards.append(episode.steps[td_index].reward)
                    else:
                        target_values.append(0.0)
                        target_rewards.append(0.0)

                policy = episode.steps[index].policy
                done = episode.steps[index].done

                sampled_data.append((obs, action, target_values, target_rewards, policy, done))
        return sampled_data
