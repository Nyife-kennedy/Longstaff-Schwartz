"""
Least-Squares Monte Carlo (LSM) for American Put Options
_weighted_laguerre_basis must normalise X by K (divide stock price by
    strike) before building the Laguerre basis.  
"""

import numpy as np
from numpy.polynomial.laguerre import lagvander


class LeastSquareMonteCarlo:
    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        No_paths: int,
        No_steps: int,
        degree: int,
    ) -> None:
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.r = r
        self.T = T
        self.No_paths = No_paths
        self.No_steps = No_steps
        self.degree = degree
        self.dt = T / No_steps
        self._paths = None

    def simulate_paths(self) -> np.ndarray:
        Z = np.random.normal(0, 1, size=(self.No_paths, self.No_steps))
        increments = np.exp(
            (self.r - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )
        S = np.empty((self.No_paths, self.No_steps + 1))
        S[:, 0] = self.S0
        S[:, 1:] = self.S0 * np.cumprod(increments, axis=1)
        return S

    @property
    def paths(self) -> np.ndarray:
        if self._paths is None:
            self._paths = self.simulate_paths()
        return self._paths

    def _weighted_laguerre_basis(self, X: np.ndarray) -> np.ndarray:
        # Normalise by strike (paper Section 8.3) before applying exp(-x/2)
        X_norm = X / self.K
        weight  = np.exp(-X_norm / 2.0)
        Phi_raw = lagvander(X_norm, self.degree)
        return Phi_raw * weight[:, np.newaxis]

    def _fit_continuation(self, X_itm: np.ndarray, Y_itm: np.ndarray) -> np.ndarray:
        Phi = self._weighted_laguerre_basis(X_itm)
        coeffs, _, _, _ = np.linalg.lstsq(Phi, Y_itm, rcond=None)
        return Phi @ coeffs

    def price(self) -> float:
        S = self.paths
        cf_matrix = np.zeros((self.No_paths, self.No_steps + 1))
        cf_matrix[:, self.No_steps] = np.maximum(self.K - S[:, self.No_steps], 0.0)

        for t in range(self.No_steps - 1, 0, -1):
            stock_t     = S[:, t]
            intrinsic_t = np.maximum(self.K - stock_t, 0.0)
            itm_mask    = intrinsic_t > 0.0
            if itm_mask.sum() == 0:
                continue

            X_itm            = stock_t[itm_mask]
            steps_forward     = np.arange(1, self.No_steps + 1 - t)
            discount_factors  = np.exp(-self.r * self.dt * steps_forward)
            future_cfs_itm    = cf_matrix[
                np.ix_(itm_mask, np.arange(t + 1, self.No_steps + 1))
            ]
            Y_itm        = future_cfs_itm @ discount_factors
            continuation = self._fit_continuation(X_itm, Y_itm)

            exercise_now     = intrinsic_t[itm_mask] >= continuation
            exercise_indices = np.where(itm_mask)[0][exercise_now]
            cf_matrix[exercise_indices, t + 1:] = 0.0
            cf_matrix[exercise_indices, t]       = intrinsic_t[exercise_indices]

        time_indices     = np.arange(self.No_steps + 1)
        discount_to_zero = np.exp(-self.r * self.dt * time_indices)
        pv_per_path      = cf_matrix @ discount_to_zero
        return float(pv_per_path.mean())

if __name__ == "__main__":
    

    S0     = 44.0
    K      = 40.0
    r      = 0.06
    sigma  = 0.20
    T      = 1
    No_paths  = 100_000   
    No_steps  = 50     
    degree    = 3         

    model = LeastSquareMonteCarlo(S0, K, r, sigma, T, No_paths, No_steps, degree)
    price = model.price()
    print(f"LSM American Put Price : {price:.4f}")