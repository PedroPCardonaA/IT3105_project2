from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol
import jax
import jax.numpy as jnp
import random


class MuZeroNetwork(Protocol):
    def pred(self, hidden: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    def recurrent_inference(
        self, hidden: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

@dataclass(frozen=True)
class MCTSconfig:
    num_simulations: int = 50
    max_depth: int = 5
    discount: float = 0.997
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    dirichlet_alpha: float = 0.25
    root_exploration_frac: float = 0.25

class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "reward", "hidden", "children", "is_expanded")

    def __init__(self, prior: float, reward: float = 0.0, hidden: Optional[jnp.ndarray] = None):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = reward
        self.hidden = hidden
        self.children: Dict[int, MCTSNode] = {}
        self.is_expanded = False

    def expanded(self) -> bool:
        return self.is_expanded
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    

def softmax_sample(logits: jnp.ndarray) -> jnp.ndarray:
    x = logits - jnp.max(logits)
    e = jnp.exp(x)
    return e / jnp.sum(e)

class MuZeroMCTS:
    def __init__(self, config: MCTSconfig, num_actions: int, network: MuZeroNetwork):

        self.config = config
        self.num_actions = num_actions
        self.network = network

    def run(self, root_hidden: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        root = MCTSNode(prior=1.0, hidden=root_hidden)
        self._expand(root, add_dirichlet=True)

        for _ in range(self.config.num_simulations):
            node = root
            search_path: List[Tuple[MCTSNode, int]] = []
            depth = 0

            # Selection: traverse tree until we find a leaf
            while node.expanded() and depth < self.config.max_depth:
                action, child = self._select_child(node)
                search_path.append((node, action))
                
                # If child doesn't have hidden state, compute it via dynamics
                if child.hidden is None and node.hidden is not None:
                    next_hidden, reward, _, _ = self.network.recurrent_inference(
                        node.hidden, jnp.array(action)
                    )
                    child.hidden = next_hidden
                    child.reward = float(reward)
                
                node = child
                depth += 1

            # If we hit max depth, evaluate and backpropagate
            if depth >= self.config.max_depth:
                value = self._evaluate(node)
                self._backpropagate(search_path, value)
                continue

            # Expand the leaf node if it hasn't been expanded
            if not node.expanded():
                self._expand(node, add_dirichlet=False)

            # Evaluate the leaf
            value = self._evaluate(node)
            self._backpropagate(search_path, value)

        policy = self._policy_from_visits(root)
        return policy, root.value()
    
    def _evaluate(self, node: MCTSNode) -> float:
        if node.hidden is None:
            return 0.0
        _, value = self.network.pred(node.hidden)
        return float(value)
    
    def _expand(self, node: MCTSNode, add_dirichlet: bool) -> None:
        
        if node.hidden is None:
            return

        policy_logits, value = self.network.pred(node.hidden)
        policy = softmax_sample(policy_logits)

        if add_dirichlet:
            alpha = self.config.dirichlet_alpha
            dirichlet_noise = jax.random.dirichlet(jax.random.PRNGKey(random.randint(0, 1 << 31)), alpha * jnp.ones(self.num_actions))
            policy = (1 - self.config.root_exploration_frac) * policy + self.config.root_exploration_frac * dirichlet_noise

        for action in range(self.num_actions):
            node.children[action] = MCTSNode(prior=float(policy[action]))

        node.is_expanded = True

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        total_visit_count = sum(child.visit_count for child in node.children.values())
        max_ucb = -float('inf')
        selected_action = -1
        selected_child = None

        for action, child in node.children.items():
            ucb_score = self._ucb_score(node, child, total_visit_count)
            if ucb_score > max_ucb:
                max_ucb = ucb_score
                selected_action = action
                selected_child = child

        assert selected_child is not None
        return selected_action, selected_child
    
    def _ucb_score(self, parent: MCTSNode, child: MCTSNode, total_visit_count: int) -> float:
        """Compute PUCT score: Q(s,a) + U(s,a) where U includes prior and exploration."""
        # Exploration term with adaptive UCB constant
        pb_c = jnp.log((total_visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= jnp.sqrt(total_visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        # Q-value: mean value from this action
        q_value = child.value()

        return float(q_value + prior_score)

    def _rollout(self, hidden: jnp.ndarray, depth: int) -> float:
        if depth == 0:
            _, value = self.network.pred(hidden)
            return float(value)

        action = random.randint(0, self.num_actions - 1)
        next_hidden, reward, _, _ = self.network.recurrent_inference(hidden, jnp.array(action))
        value_from_next = self._rollout(next_hidden, depth - 1)
        return float(reward) + self.config.discount * value_from_next
    
    def _backpropagate(self, search_path: List[Tuple[MCTSNode, int]], value: float) -> None:
        """Backpropagate value estimates up the search tree.
        
        At each node, we update the value estimate by incorporating the reward
        received when transitioning to the child plus the discounted value of the child.
        """
        # Update statistics along the selected path.
        # We must increment visit counts of the *children* that were selected;
        # the root policy is derived from children visit counts.
        for node, action in reversed(search_path):
            child = node.children.get(action)
            if child is None:
                value = self.config.discount * value
                continue

            child.value_sum += value
            child.visit_count += 1

            # Also track parent visits/values for diagnostics (root.value()).
            node.value_sum += value
            node.visit_count += 1

            value = child.reward + self.config.discount * value

    def _policy_from_visits(self, root: MCTSNode) -> jnp.ndarray:
        visit_counts = jnp.array(
            [root.children[a].visit_count for a in range(self.num_actions)],
            dtype=jnp.float32,
        )
        total = jnp.sum(visit_counts)

        # If the tree didn't record any visits (can happen early / with bugs),
        # fall back to priors to avoid NaNs.
        def from_priors() -> jnp.ndarray:
            priors = jnp.array([root.children[a].prior for a in range(self.num_actions)], dtype=jnp.float32)
            priors_sum = jnp.sum(priors)
            return jnp.where(
                priors_sum > 0,
                priors / priors_sum,
                jnp.ones((self.num_actions,), dtype=jnp.float32) / self.num_actions,
            )

        return jnp.where(total > 0, visit_counts / total, from_priors())
