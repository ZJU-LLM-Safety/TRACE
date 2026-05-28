from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

import logging

logger = logging.getLogger("execute_step")


ACTIONS: List[str] = [
    "role_generalize",
    "role_operationalize",
    "role_replace",
    "env_emphasize_tool_usage",
    "env_tighten_scope",
    "env_replace",
    "directive_strengthen_constraints",
    "directive_shorten",
    "directive_replace",
    "directive_make_stepwise",
    "tips_strengthen",
    "tips_concretize",
    "tips_reorder",
    "tips_prune",
    "tips_replace_one",
]

START_STATE = "init"


@dataclass
class ActionPolicy:
    """
    Q-learning based first-order Markov transition model over actions.

    State:
        last_action
    Action:
        next_action

    Q[last_action, next_action] represents the utility of choosing
    `next_action` given the previous action `last_action`.
    """
    actions: List[str]
    alpha: float = 0.2
    gamma: float = 0.9
    # epsilon: float = 0.1
    init_q: float = 1.0
    temperature: float = 1.5
    min_temp: float = 0.5
    max_temp: float = 2
    temp_decay: float = 0.99
    lam: float = 0.05
    collapse_ratio: float = 0.4
    boost_factor: float = 1.5
    update_step: int = 0
    seed: Optional[int] = None

    action_to_idx: Dict[str, int] = field(init=False)
    # idx_to_action: Dict[int, str] = field(init=False)
    action_visits: np.ndarray = field(init=False)
    action_temp: np.ndarray = field(init=False)
    q_table: np.ndarray = field(init=False)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.action_to_idx = {a: i for i, a in enumerate(self.actions + [START_STATE])}
        # self.idx_to_action = {i: a for i, a in enumerate(self.actions)}
        n = len(self.actions)
        self.action_visits = np.zeros(n, dtype=np.int32)
        self.action_temp = np.full(n, fill_value=self.temperature, dtype=np.float32)
        self.q_table = np.full((n + 1, n), fill_value=self.init_q, dtype=np.float32)
        # self.rng = random.Random(self.seed)
        self.rng = random.Random()

    def _check_action(self, action: str) -> None:
        if action not in self.action_to_idx:
            raise ValueError(f"Unknown action: {action}")

    def get_q(self, last_action: str, next_action: str) -> float:
        self._check_action(last_action)
        self._check_action(next_action)
        i = self.action_to_idx[last_action]
        j = self.action_to_idx[next_action]
        return float(self.q_table[i, j])

    def get_row(self, last_action: str) -> Dict[str, float]:
        self._check_action(last_action)
        i = self.action_to_idx[last_action]
        return {
            self.idx_to_action[j]: float(self.q_table[i, j])
            for j in range(len(self.actions))
        }

    def adaptive_temperature(
        self,
        base_temp,
        q_values,
        # collapse_ratio=0.4,
        # boost_factor=1.5,
    ):    
        q_values = np.asarray(q_values, dtype=np.float64)

        # softmax
        logits = q_values / base_temp
        logits = logits - np.max(logits)   # 数值稳定
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        # entropy
        safe_probs = probs + 1e-12
        H = -np.sum(safe_probs * np.log(safe_probs))

        # collapse
        H_max = np.log(len(q_values))
        threshold = self.collapse_ratio * H_max

        if H < threshold:
            new_temp = min(base_temp * self.boost_factor, self.max_temp)
        else:
            new_temp = base_temp

        new_temp = max(self.min_temp, new_temp)

        return new_temp


    def select_next_action(
        self,
        last_action: str,
        valid_next_actions: Optional[List[str]] = None,
        # epsilon: Optional[float] = None,
    ) -> str:
        """
        Epsilon-greedy selection of next action.
        """
        self._check_action(last_action)
        # eps = self.epsilon if epsilon is None else epsilon

        if valid_next_actions is None:
            candidate_actions = self.actions
        else:
            for a in valid_next_actions:
                self._check_action(a)
            if not valid_next_actions:
                raise ValueError("valid_next_actions must not be empty.")
            candidate_actions = valid_next_actions

            # randomly explore with probability eps
            # if self.rng.random() < eps:
            #     return self.rng.choice(candidate_actions)

        if last_action == START_STATE:
            return self.select_initial_action(candidate_actions)

        i = self.action_to_idx[last_action]
        self.action_visits[i] += 1
        candidate_indices = [self.action_to_idx[a] for a in candidate_actions]
        q_values = np.array([self.q_table[i, j] for j in candidate_indices], dtype=np.float64)

        self.action_temp[i] = self.adaptive_temperature(
            base_temp=self.action_temp[i],
            q_values=q_values,
        )
        # Softmax stabilization
        q_values = q_values / self.action_temp[i]
        q_values = q_values - np.max(q_values)

        # temperature decay
        self.action_temp[i] = max(self.min_temp, self.action_temp[i] * self.temp_decay)

        exp_q = np.exp(q_values)
        probs = exp_q / np.sum(exp_q)

        chosen_idx = self.rng.choices(range(len(candidate_actions)), weights=probs, k=1)[0]

        return candidate_actions[chosen_idx]
        # candidate_qs = [self.q_table[i, j] for j in candidate_indices]
        # max_q = max(candidate_qs)

        # best_actions = [
        #     self.idx_to_action[j]
        #     for j in candidate_indices
        #     if self.q_table[i, j] == max_q
        # ]
        # return self.rng.choice(best_actions)

    def select_initial_action(self, candidate_actions) -> str:
        # First step has no previous action; aggregate utility per next action
        # across all possible previous actions, then sample by softmax.
        # q_sums = np.sum(self.q_table, axis=0, dtype=np.float64)

        # logits = q_sums / self.temperature
        # logits = logits - np.max(logits)
        # exp_logits = np.exp(logits)
        # probs = exp_logits / np.sum(exp_logits)

        # chosen_idx = self.rng.choices(range(len(self.actions)), weights=probs, k=1)[0]
        # return self.actions[chosen_idx]
        i = self.action_to_idx[START_STATE]
        candidate_indices = [self.action_to_idx[a] for a in candidate_actions]
        q_values = np.array([self.q_table[i, j] for j in candidate_indices], dtype=np.float64)
        # Softmax stabilization
        q_values = q_values / self.temperature
        q_values = q_values - np.max(q_values)

        exp_q = np.exp(q_values)
        probs = exp_q / np.sum(exp_q)
        logger.debug(f"selection probability: {probs}")
        chosen_idx = self.rng.choices(range(len(candidate_actions)), weights=probs, k=1)[0]

        return candidate_actions[chosen_idx]

    def update(
        self,
        last_action: str,
        chosen_next_action: str,
        reward: float,
        new_last_action: Optional[str] = None,
        valid_future_actions: Optional[List[str]] = None,
        done: bool = False,
    ) -> None:
        """
        Standard Q-learning update:

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

        Here:
            s  = last_action
            a  = chosen_next_action
            s' = new_last_action

        Usually in this setting:
            new_last_action == chosen_next_action
        because once you execute the chosen next action, it becomes the
        previous action for the next decision step.
        """
        self._check_action(last_action)
        self._check_action(chosen_next_action)

        if new_last_action is None:
            new_last_action = chosen_next_action
        self._check_action(new_last_action)

        i = self.action_to_idx[last_action]
        j = self.action_to_idx[chosen_next_action]

        current_q = self.q_table[i, j]
        reward /= 2
        if done:
            td_target = reward
        else:
            s_next = self.action_to_idx[new_last_action]

            if valid_future_actions is None:
                future_indices = list(range(len(self.actions)))
            else:
                for a in valid_future_actions:
                    self._check_action(a)
                if not valid_future_actions:
                    raise ValueError("valid_future_actions must not be empty.")
                future_indices = [self.action_to_idx[a] for a in valid_future_actions]

            max_next_q = max(float(self.q_table[s_next, k]) for k in future_indices)
            td_target = reward + self.gamma * max_next_q

        self.q_table[i, j] = current_q + self.alpha * (td_target - current_q)
        self.update_step += 1
        if self.update_step % 50 == 0:
            logger.debug(f"self.lam: {self.lam}")
            self.q_table = type(self).shrink_q_table_toward_uniform(q_table=self.q_table, lam=self.lam)
            self.update_step = 0

    def transition_probs(
        self,
        last_action: str,
        temperature: float = 1.0,
        valid_next_actions: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Convert Q row into a softmax probability distribution.
        Useful if you want a stochastic Markov transition matrix.
        """
        self._check_action(last_action)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        i = self.action_to_idx[last_action]

        if valid_next_actions is None:
            candidate_actions = self.actions
        else:
            for a in valid_next_actions:
                self._check_action(a)
            if not valid_next_actions:
                raise ValueError("valid_next_actions must not be empty.")
            candidate_actions = valid_next_actions

        candidate_indices = [self.action_to_idx[a] for a in candidate_actions]
        q_values = np.array([self.q_table[i, j] for j in candidate_indices], dtype=np.float64)

        # Softmax stabilization
        q_values = q_values / temperature
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values)
        probs = exp_q / np.sum(exp_q)

        return {
            action: float(prob)
            for action, prob in zip(candidate_actions, probs)
        }

    def greedy_transition_matrix(self) -> np.ndarray:
        """
        Return row-wise greedy one-hot transition matrix derived from Q.
        Each row picks argmax next action.
        """
        n = len(self.actions)
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            row = self.q_table[i]
            max_q = np.max(row)
            best_indices = np.flatnonzero(row == max_q)
            chosen = int(best_indices[0])
            mat[i, chosen] = 1.0
        return mat

    def soft_transition_matrix(self, temperature: float = 1.0) -> np.ndarray:
        """
        Return row-wise softmax transition matrix derived from Q.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        q = self.q_table.astype(np.float64) / temperature
        q = q - np.max(q, axis=1, keepdims=True)
        exp_q = np.exp(q)
        probs = exp_q / np.sum(exp_q, axis=1, keepdims=True)
        return probs.astype(np.float32)

    def pretty_print_q(self, decimals: int = 3) -> None:
        header = ["last\\next"] + self.actions
        print("\t".join(header))
        for i, a in enumerate(self.actions):
            row = [a] + [f"{self.q_table[i, j]:.{decimals}f}" for j in range(len(self.actions))]
            print("\t".join(row))

    # TODO step % 50 == 0
    @staticmethod
    def shrink_q_table_toward_uniform(q_table: np.ndarray, lam: float = 0.05) -> np.ndarray:
        row_means = np.mean(q_table, axis=1, keepdims=True)
        return (1 - lam) * q_table + lam * row_means

    def save(self, file_path: str) -> None:
        """
        Persist the model to a compressed .npz file.
        """
        np.savez_compressed(
            file_path,
            actions=np.array(self.actions, dtype=object),
            alpha=np.array(self.alpha, dtype=np.float64),
            gamma=np.array(self.gamma, dtype=np.float64),
            init_q=np.array(self.init_q, dtype=np.float64),
            temperature=np.array(self.temperature, dtype=np.float64),
            min_temp=np.array(self.min_temp, dtype=np.float64),
            max_temp=np.array(self.max_temp, dtype=np.float64),
            temp_decay=np.array(self.temp_decay, dtype=np.float64),
            lam=np.array(self.lam, dtype=np.float64),
            collapse_ratio=np.array(self.collapse_ratio, dtype=np.float64),
            boost_factor=np.array(self.boost_factor, dtype=np.float64),
            update_step=np.array(self.update_step, dtype=np.int64),
            seed=np.array(self.seed, dtype=object),
            action_visits=self.action_visits.astype(np.int32),
            action_temp=self.action_temp.astype(np.float32),
            q_table=self.q_table.astype(np.float32),
        )

    @classmethod
    def load(cls, file_path: str) -> "ActionPolicy":
        """
        Load a model from a file previously saved by `save`.
        """
        with np.load(file_path, allow_pickle=True) as data:
            actions = [str(a) for a in data["actions"].tolist()]
            alpha = float(data["alpha"])
            gamma = float(data["gamma"])
            init_q = float(data["init_q"])
            temperature = float(data["temperature"])
            min_temp = float(data["min_temp"]) if "min_temp" in data else 0.5
            max_temp = float(data["max_temp"]) if "max_temp" in data else 2.0
            temp_decay = float(data["temp_decay"]) if "temp_decay" in data else 0.99
            lam = float(np.asarray(data["lam"]).reshape(-1)[0]) if "lam" in data else 0.05
            collapse_ratio = float(np.asarray(data["collapse_ratio"]).reshape(-1)[0]) if "collapse_ratio" in data else 0.4
            boost_factor = float(np.asarray(data["boost_factor"]).reshape(-1)[0]) if "boost_factor" in data else 1.5
            update_step = int(np.asarray(data["update_step"]).reshape(-1)[0]) if "update_step" in data else 0

            seed_obj = data["seed"].item()
            seed = None if seed_obj is None else int(seed_obj)

            q_table = np.array(data["q_table"], dtype=np.float32)
            action_visits = (
                np.array(data["action_visits"], dtype=np.int32)
                if "action_visits" in data
                else None
            )
            action_temp = (
                np.array(data["action_temp"], dtype=np.float32)
                if "action_temp" in data
                else None
            )

        n = len(actions)
        if q_table.shape != (n + 1, n):
            raise ValueError(
                f"Invalid q_table shape {q_table.shape}, expected {(n + 1, n)}."
            )

        model = cls(
            actions=actions,
            alpha=alpha,
            gamma=gamma,
            init_q=init_q,
            temperature=temperature,
            min_temp=min_temp,
            max_temp=max_temp,
            temp_decay=temp_decay,
            lam=lam,
            collapse_ratio=collapse_ratio,
            boost_factor=boost_factor,
            update_step=update_step,
            seed=seed,
        )
        model.q_table = q_table
        if action_visits is not None and action_visits.shape == (n,):
            model.action_visits = action_visits
        if action_temp is not None and action_temp.shape == (n,):
            model.action_temp = action_temp
        return model

def build_default_model(
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    init_q: float = 0.0,
    seed: Optional[int] = 42,
) -> ActionPolicy:
    return ActionPolicy(
        actions=ACTIONS,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        init_q=init_q,
        seed=seed,
    )


if __name__ == "__main__":
    model = build_default_model(
        alpha=0.2,
        gamma=0.9,
        epsilon=0.15,
        init_q=1.0,   # all elements initialized equally
        seed=123,
        temperature=1.5
    )

    # Example:
    # last action = role_generalize
    last_action = "role_generalize"

    # choose next action by epsilon-greedy
    next_action = model.select_next_action(last_action)
    print("Selected next action:", next_action)

    # suppose executing next_action gets reward = 0.8
    reward = 0.8

    # Q-learning update
    # after executing next_action, it becomes the new "last_action"
    model.update(
        last_action=last_action,
        chosen_next_action=next_action,
        reward=reward,
        new_last_action=next_action,
        done=False,
    )

    # inspect Q values for one row
    print("\nQ row for role_generalize:")
    print(model.get_row("role_generalize"))

    # derive stochastic Markov transition probabilities from Q
    print("\nSoftmax transition probs from role_generalize:")
    print(model.transition_probs("role_generalize", temperature=0.5))

    # full soft transition matrix
    soft_mat = model.soft_transition_matrix(temperature=0.5)
    print("\nSoft transition matrix shape:", soft_mat.shape)

    # pretty print Q table
    print("\nCurrent Q table:")
    model.pretty_print_q()
