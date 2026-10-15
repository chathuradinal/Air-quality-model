"""
Dung Beetle Optimizer (DBO) Module.
Faithfully implements the meta-heuristic algorithm by Xue & Shen (2022) as cited in Duan et al. (2023).
Includes all 4 biological behaviors:
1. Ball Rolling & Dancing (Eqs. 10, 11)
2. Breeding / Spawning Area (Eqs. 12, 13)
3. Foraging / Little Beetles (Eqs. 14, 15)
4. Stealing / Thieving Beetles (Eq. 16)
"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Any


class DungBeetleOptimizer:
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        pop_size: int = 20,
        max_iter: int = 15,
        seed: int = 42,
    ):
        """
        Args:
            objective_func: Function accepting a position vector x of shape (dim,) and returning fitness (e.g., RMSE).
            lb: Lower bounds for each dimension (dim,).
            ub: Upper bounds for each dimension (dim,).
            dim: Dimension of optimization problem.
            pop_size: Total population size N (default 20).
            max_iter: Maximum iterations T (default 15).
            seed: Random seed for reproducibility.
        """
        self.objective_func = objective_func
        self.lb = np.array(lb, dtype=np.float64)
        self.ub = np.array(ub, dtype=np.float64)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.RandomState(seed)

        # Proportions of the 4 population groups (standard DBO literature)
        # N1: Ball-rolling beetles (~60%)
        # N2: Breeding eggs (~20%)
        # N3: Foraging small beetles (~10%)
        # N4: Thieving beetles (~10%)
        self.n1 = max(1, int(0.6 * self.pop_size))
        self.n2 = max(1, int(0.2 * self.pop_size))
        self.n3 = max(1, int(0.1 * self.pop_size))
        self.n4 = max(1, self.pop_size - self.n1 - self.n2 - self.n3)

        # Positions and fitness tracking
        self.positions = np.zeros((self.pop_size, self.dim), dtype=np.float64)
        self.prev_positions = np.zeros((self.pop_size, self.dim), dtype=np.float64)
        self.fitness = np.full(self.pop_size, float("inf"), dtype=np.float64)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.worst_solution = None
        self.convergence_curve = []

    def _init_population(self):
        """Uniformly initializes population within bounds [lb, ub]."""
        for i in range(self.pop_size):
            self.positions[i] = self.lb + self.rng.rand(self.dim) * (self.ub - self.lb)
            self.prev_positions[i] = np.copy(self.positions[i])
            self.fitness[i] = self.objective_func(self.positions[i])

        self._update_global_best_and_worst()

    def _update_global_best_and_worst(self):
        min_idx = np.argmin(self.fitness)
        max_idx = np.argmax(self.fitness)

        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[min_idx])
            self.best_solution = np.copy(self.positions[min_idx])

        self.worst_solution = np.copy(self.positions[max_idx])

    def _clip_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """Executes full DBO optimization loop."""
        print(f"Initializing DBO Population (N={self.pop_size}, T={self.max_iter}, D={self.dim})...")
        self._init_population()
        self.convergence_curve.append(self.best_fitness)
        print(f"Iter 0: Initial Best Fitness = {self.best_fitness:.4f}")

        k = 0.1  # deflection coefficient in (0, 0.2]
        b = 0.3  # constant in (0, 1)
        S = 0.5  # constant value for stealing

        for t in range(1, self.max_iter + 1):
            R = 1.0 - (t / self.max_iter)
            curr_positions = np.copy(self.positions)

            # 1. Ball Rolling & Dancing Phase (Indices 0 to n1)
            for i in range(self.n1):
                prob = self.rng.rand()
                if prob < 0.9:
                    # Unobstructed ball rolling
                    alpha = 1.0 if self.rng.rand() > 0.5 else -1.0
                    delta_x = np.abs(curr_positions[i] - self.worst_solution)
                    self.positions[i] = (
                        curr_positions[i]
                        + alpha * k * self.prev_positions[i]
                        + b * delta_x
                    )
                else:
                    # Dancing behavior (obstacle encounter)
                    theta = self.rng.uniform(0, np.pi)
                    if not np.isclose(theta, np.pi / 2, atol=1e-2):
                        self.positions[i] = (
                            curr_positions[i]
                            + np.tan(theta) * np.abs(curr_positions[i] - self.prev_positions[i])
                        )
                self.positions[i] = self._clip_bounds(self.positions[i])

            # 2. Breeding / Egg-Laying Phase (Indices n1 to n1 + n2)
            # Spawning boundaries Lb*, Ub* (Eq. 12)
            bound1 = self.best_solution * (1.0 - R)
            bound2 = self.best_solution * (1.0 + R)
            lb_star = np.maximum(np.minimum(bound1, bound2), self.lb)
            ub_star = np.minimum(np.maximum(bound1, bound2), self.ub)

            for i in range(self.n1, self.n1 + self.n2):
                b1 = self.rng.rand(self.dim)
                b2 = self.rng.rand(self.dim)
                self.positions[i] = (
                    self.best_solution
                    + b1 * (curr_positions[i] - lb_star)
                    + b2 * (curr_positions[i] - ub_star)
                )
                self.positions[i] = self._clip_bounds(self.positions[i])

            # 3. Foraging Phase (Indices n1 + n2 to n1 + n2 + n3)
            # Optimal foraging boundaries Lbb, Ubb (Eq. 14)
            bound_b1 = self.best_solution * (1.0 - R)
            bound_b2 = self.best_solution * (1.0 + R)
            lbb = np.maximum(np.minimum(bound_b1, bound_b2), self.lb)
            ubb = np.minimum(np.maximum(bound_b1, bound_b2), self.ub)

            for i in range(self.n1 + self.n2, self.n1 + self.n2 + self.n3):
                c1 = self.rng.randn(self.dim)
                c2 = self.rng.rand(self.dim)
                self.positions[i] = (
                    curr_positions[i]
                    + c1 * (curr_positions[i] - lbb)
                    + c2 * (curr_positions[i] - ubb)
                )
                self.positions[i] = self._clip_bounds(self.positions[i])

            # 4. Stealing Phase (Indices n1 + n2 + n3 to pop_size)
            for i in range(self.n1 + self.n2 + self.n3, self.pop_size):
                g = self.rng.randn(self.dim)
                self.positions[i] = (
                    self.best_solution
                    + S * g * (np.abs(curr_positions[i] - self.best_solution)
                               + np.abs(curr_positions[i] - self.worst_solution))
                )
                self.positions[i] = self._clip_bounds(self.positions[i])

            # Evaluate new positions and retain if improved (greedy selection)
            for i in range(self.pop_size):
                fit_val = self.objective_func(self.positions[i])
                if fit_val < self.fitness[i]:
                    self.fitness[i] = fit_val
                    self.prev_positions[i] = np.copy(curr_positions[i])
                else:
                    self.positions[i] = np.copy(curr_positions[i])

            self._update_global_best_and_worst()
            self.convergence_curve.append(self.best_fitness)
            print(f"Iter {t}/{self.max_iter} - Best Fitness: {self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness, self.convergence_curve


def decode_hyperparameters(vector: np.ndarray) -> Dict[str, Any]:
    """
    Decodes continuous DBO vector into discrete & continuous CNN-LSTM hyperparameters.
    Vector structure (8 dimensions):
    [0]: sliding_window in [5, 30]
    [1]: lstm_layer1 in [20, 200]
    [2]: lstm_layer2 in [20, 200]
    [3]: lstm_layer3 in [20, 200]
    [4]: filters in [32, 256]
    [5]: kernel_size in [2, 5]
    [6]: strides in [1, 2]
    [7]: learning_rate in [0.005, 0.1]
    """
    return {
        "sliding_window": int(round(vector[0])),
        "lstm_neurons": (
            int(round(vector[1])),
            int(round(vector[2])),
            int(round(vector[3])),
        ),
        "filters": int(round(vector[4])),
        "kernel_size": int(round(vector[5])),
        "strides": int(round(vector[6])),
        "learning_rate": float(vector[7]),
        "batch_size": 64,
        "epochs": 40,
    }
