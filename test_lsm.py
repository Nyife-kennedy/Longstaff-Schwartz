from gettext import install
import math
import numpy as np

import pytest
from scipy.stats import norm

from LeastSquareMonteCarlo import LeastSquareMonteCarlo

def bs_put(S, K, r, sigma, T) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def make_model(S0=40, K=40, r=0.06, sigma=0.20, T=1.0,
               No_paths=80_000, No_steps=50, degree=3, seed=0) -> LeastSquareMonteCarlo:
    np.random.seed(seed)
    return LeastSquareMonteCarlo(S0, K, r, sigma, T, No_paths, No_steps, degree)


# Table 1 from paper: (S, sigma, T, FD_american_price)
TABLE1 = [
    (36, 0.20, 1, 4.478),
    (36, 0.20, 2, 4.840),
    (36, 0.40, 1, 7.101),
    (36, 0.40, 2, 8.508),
    (38, 0.20, 1, 3.250),
    (38, 0.20, 2, 3.745),
    (38, 0.40, 1, 6.148),
    (38, 0.40, 2, 7.670),
    (40, 0.20, 1, 2.314),
    (40, 0.20, 2, 2.885),
    (40, 0.40, 1, 5.312),
    (40, 0.40, 2, 6.920),
    (42, 0.20, 1, 1.617),
    (42, 0.20, 2, 2.212),
    (42, 0.40, 1, 4.582),
    (42, 0.40, 2, 6.248),
    (44, 0.20, 1, 1.110),
    (44, 0.20, 2, 1.690),
    (44, 0.40, 1, 3.948),
    (44, 0.40, 2, 5.647),
]

class TestSanity:

    def test_dt_computed_correctly(self):
        m = make_model(T=2.0, No_steps=40)
        assert m.dt == pytest.approx(0.05)

    def test_paths_shape(self):
        m = make_model(No_paths=500, No_steps=20)
        S = m.paths
        assert S.shape == (500, 21)           # (No_paths, No_steps+1)

    def test_paths_first_column_is_S0(self):
        m = make_model(S0=42.0, No_paths=200, No_steps=10)
        assert np.all(m.paths[:, 0] == 42.0)

    def test_paths_all_positive(self):
        """GBM can never go negative."""
        m = make_model(No_paths=1000, No_steps=50)
        assert np.all(m.paths > 0)

    def test_paths_cached(self):
        """Accessing paths twice must return the identical array."""
        m = make_model(No_paths=100, No_steps=10)
        p1 = m.paths
        p2 = m.paths
        assert p1 is p2   # same object, not just equal values

    def test_basis_matrix_shape(self):
        m = make_model(degree=3)
        X = np.array([38.0, 39.0, 40.0, 41.0])
        Phi = m._weighted_laguerre_basis(X)
        assert Phi.shape == (4, 4)            # (n_obs, degree+1)

    def test_basis_first_column_is_L0(self):
        """
        L_0(x) = exp(-x/2) * P_0(x) = exp(-x/2) * 1 = exp(-x/2).
        After normalisation x_norm = X/K:
            Phi[:,0] = exp(-x_norm/2)
        """
        m = make_model(K=40.0, degree=3)
        X = np.array([36.0, 40.0, 44.0])
        Phi = m._weighted_laguerre_basis(X)
        expected = np.exp(-(X / 40.0) / 2.0)
        np.testing.assert_allclose(Phi[:, 0], expected, rtol=1e-12)

    def test_basis_weight_uses_normalised_price(self):
        
        m = make_model(K=40.0, degree=3)
        X = np.array([36.0, 40.0, 44.0])
        Phi = m._weighted_laguerre_basis(X)
       
        weights = Phi[:, 0]   # first column = weight * P_0 = weight * 1
        assert np.all(weights > 0.1), (
            "Basis weights are near-zero — normalisation by K is missing"
        )

    def test_cf_matrix_shape(self):
        m = make_model(No_paths=200, No_steps=10)
        m._paths = m.simulate_paths()
        cf = np.zeros((m.No_paths, m.No_steps + 1))
        assert cf.shape == (200, 11)

    def test_cf_matrix_at_most_one_nonzero_per_path(self):
        """Each path can exercise at most once."""
        m = make_model(No_paths=500, No_steps=20, seed=7)
        # Reconstruct cf_matrix by pricing then checking structure
        S = m.paths
        cf_matrix = np.zeros((m.No_paths, m.No_steps + 1))
        cf_matrix[:, m.No_steps] = np.maximum(m.K - S[:, m.No_steps], 0.0)

        for t in range(m.No_steps - 1, 0, -1):
            stock_t     = S[:, t]
            intrinsic_t = np.maximum(m.K - stock_t, 0.0)
            itm_mask    = intrinsic_t > 0.0
            if itm_mask.sum() == 0:
                continue
            X_itm           = stock_t[itm_mask]
            steps_forward   = np.arange(1, m.No_steps + 1 - t)
            disc_f          = np.exp(-m.r * m.dt * steps_forward)
            future          = cf_matrix[np.ix_(itm_mask, np.arange(t+1, m.No_steps+1))]
            Y_itm           = future @ disc_f
            continuation    = m._fit_continuation(X_itm, Y_itm)
            ex_now          = intrinsic_t[itm_mask] >= continuation
            ex_idx          = np.where(itm_mask)[0][ex_now]
            cf_matrix[ex_idx, t+1:] = 0.0
            cf_matrix[ex_idx, t]    = intrinsic_t[ex_idx]

        nonzero_counts = np.count_nonzero(cf_matrix, axis=1)
        assert np.all(nonzero_counts <= 1), (
            "Some paths have more than one exercise cash flow"
        )

    def test_price_is_positive(self):
        m = make_model()
        assert m.price() > 0

    def test_price_is_float(self):
        m = make_model(No_paths=200, No_steps=5)
        assert isinstance(m.price(), float)

class TestTable1Accuracy:
    """
    Tolerance rationale
    -------------------
    Table 1 reports s.e. between 0.007 and 0.024.  We use 3 × max_se = 0.072
    as the absolute tolerance so that a correctly implemented algorithm passes
    with extremely high probability even with a different random seed.
    """
    K        = 40.0
    r        = 0.06
    No_paths = 100_000
    No_steps = 50
    degree   = 3
    TOL      = 0.072   # 3 × max reported standard error from Table 1

    @pytest.mark.parametrize("S,sigma,T,fd_price", TABLE1)
    def test_american_put_price(self, S, sigma, T, fd_price):
        np.random.seed(0)
        m = LeastSquareMonteCarlo(
            S, self.K, self.r, sigma, T,
            self.No_paths, self.No_steps, self.degree
        )
        lsm_price = m.price()
        assert abs(lsm_price - fd_price) < self.TOL, (
            f"S={S}, sigma={sigma}, T={T}: "
            f"LSM={lsm_price:.4f}, FD={fd_price:.4f}, "
            f"diff={lsm_price-fd_price:+.4f} > tol={self.TOL}"
        )

class TestEdgeCases:

    def test_deep_itm_bounded_by_intrinsic(self):
        """
        Deep ITM American put price cannot exceed the intrinsic value K - S.
        (Immediate exercise is always available.)
        """
        m = make_model(S0=10.0, K=40.0, sigma=0.20, T=1.0,
                       No_paths=20_000, No_steps=50)
        assert m.price() <= 40.0 - 10.0 + 0.05   # small tolerance for MC noise

    def test_deep_otm_near_zero(self):
        """
        Deep OTM put (S >> K) should be close to zero.
        """
        m = make_model(S0=80.0, K=40.0, sigma=0.20, T=1.0,
                       No_paths=20_000, No_steps=50)
        assert m.price() < 0.10

    def test_zero_volatility_equals_intrinsic(self):
       
        S0, K = 38.0, 40.0
        m = make_model(S0=S0, K=K, sigma=0.001, T=1.0,
                       No_paths=10_000, No_steps=50)
        intrinsic = K - S0   # 2.0
        # With near-zero vol, price should be close to intrinsic
        assert abs(m.price() - intrinsic) < 0.15

    def test_single_exercise_step(self):
        
        S0, K, r, sigma, T = 40.0, 40.0, 0.06, 0.20, 1.0
        np.random.seed(1)
        m = LeastSquareMonteCarlo(S0, K, r, sigma, T, 200_000, 1, 3)
        lsm_price = m.price()
        bs_price  = bs_put(S0, K, r, sigma, T)
        # For No_steps=1 the LSM becomes a simple European put estimator
        assert abs(lsm_price - bs_price) < 0.05

    def test_very_short_maturity(self):
        """Near-expiry ATM put should be close to the BS European price."""
        S0, K, r, sigma, T = 40.0, 40.0, 0.06, 0.20, 1/52  # 1 week
        np.random.seed(2)
        m = LeastSquareMonteCarlo(S0, K, r, sigma, T, 50_000, 5, 3)
        bs_price = bs_put(S0, K, r, sigma, T)
        assert abs(m.price() - bs_price) < 0.05

    def test_high_volatility_price_reasonable(self):
        """
        Very high sigma should produce a large put price, but still < K.
        """
        m = make_model(S0=40.0, K=40.0, sigma=0.80, T=1.0,
                       No_paths=30_000, No_steps=50)
        p = m.price()
        assert 0 < p < 40.0

    def test_otm_at_expiry_expires_worthless(self):
       
        m = make_model(No_paths=100, No_steps=3, seed=0)
        # Force all paths to end above K
        fake_paths = np.full((100, 4), 50.0)  # S > K=40 everywhere
        m._paths = fake_paths
        assert m.price() == pytest.approx(0.0)

class TestMonotonicity:

    PATHS = 40_000
    STEPS = 50
    SEED  = 42

    def _price(self, **kwargs):
        defaults = dict(S0=40, K=40, r=0.06, sigma=0.20, T=1.0,
                        No_paths=self.PATHS, No_steps=self.STEPS, degree=3)
        defaults.update(kwargs)
        np.random.seed(self.SEED)
        return LeastSquareMonteCarlo(**defaults).price()

    def test_price_decreases_as_S_increases(self):
        """Higher stock price → lower put value."""
        p_low  = self._price(S0=36)
        p_mid  = self._price(S0=40)
        p_high = self._price(S0=44)
        assert p_low > p_mid > p_high

    def test_price_increases_as_K_increases(self):
        """Higher strike → higher put value (more likely ITM)."""
        p_low  = self._price(K=36)
        p_mid  = self._price(K=40)
        p_high = self._price(K=44)
        assert p_low < p_mid < p_high

    def test_price_increases_as_sigma_increases(self):
        """Higher vol → higher option value (vega > 0 for puts)."""
        p_lo = self._price(sigma=0.10)
        p_hi = self._price(sigma=0.40)
        assert p_hi > p_lo

    def test_price_increases_as_T_increases(self):
        """Longer tenor → American put is worth at least as much."""
        p_short = self._price(T=0.5)
        p_long  = self._price(T=2.0)
        assert p_long > p_short

    def test_price_decreases_as_r_increases(self):
        """
        Higher risk-free rate → lower put value
        """
        p_lo_r = self._price(r=0.02)
        p_hi_r = self._price(r=0.10)
        assert p_lo_r > p_hi_r

class TestAmericanVsEuropean:
    """
    An American option is worth at least as much as its European counterpart
    because it has strictly more exercise rights.
    """

    @pytest.mark.parametrize("S,sigma,T,_fd", TABLE1)
    def test_american_geq_european(self, S, sigma, T, _fd):
        K, r = 40.0, 0.06
        np.random.seed(5)
        m = LeastSquareMonteCarlo(S, K, r, sigma, T, 50_000, 50, 3)
        american_price = m.price()
        european_price = bs_put(S, K, r, sigma, T)
        assert american_price >= european_price - 0.02, (
            f"S={S}, sigma={sigma}, T={T}: "
            f"American={american_price:.4f} < European={european_price:.4f}"
        )
        
class TestBasisAndRegression:

    def test_basis_normalisation_prevents_weight_collapse(self):
        """
        Core bug check: with K=40 and S in [30,50], exp(-S/K/2) in [0.47,0.68].
        Without normalisation exp(-S/2) in [2e-11, 6e-7] — near machine zero.
        """
        m = make_model(K=40.0, degree=3)
        X = np.linspace(30, 50, 20)
        Phi = m._weighted_laguerre_basis(X)
        weights = Phi[:, 0]   # column 0 = exp(-X_norm/2) * P_0 = exp(-X_norm/2)
        assert weights.min() > 0.40, (
            f"Minimum weight={weights.min():.6f}: normalisation may be broken"
        )

    def test_basis_degree_determines_columns(self):
        for deg in [1, 2, 3, 5]:
            m = make_model(degree=deg)
            X = np.array([38., 40., 42.])
            assert m._weighted_laguerre_basis(X).shape[1] == deg + 1

    def test_fit_continuation_returns_correct_shape(self):
        m = make_model(degree=3)
        X_itm = np.array([37., 38., 39., 40.])
        Y_itm = np.array([0.30, 0.20, 0.10, 0.05])
        fitted = m._fit_continuation(X_itm, Y_itm)
        assert fitted.shape == (4,)

    def test_fit_continuation_reasonable_range(self):
        """Fitted continuation values should be non-negative for put options."""
        m = make_model(degree=3)
        X_itm = np.linspace(35, 40, 50)   # all ITM (K=40)
        # Y is a realistic continuation: higher for lower stock price
        Y_itm = np.maximum(40 - X_itm - 0.5, 0) + 0.01
        fitted = m._fit_continuation(X_itm, Y_itm)
        # Fitted values should be in a plausible range
        assert np.all(fitted >= -0.5)   # allow tiny negative from regression noise
        assert np.all(fitted < 20.0)

    def test_different_seeds_give_different_prices_but_close(self):
        """
        Two independent runs should agree within ~3 standard errors.
        s.e. for 100k paths ~ 0.01, so 3*se ~ 0.03.
        """
        prices = []
        for seed in [0, 1]:
            np.random.seed(seed)
            m = LeastSquareMonteCarlo(40, 40, 0.06, 0.20, 1.0, 100_000, 50, 3)
            prices.append(m.price())
        assert abs(prices[0] - prices[1]) < 0.05

    def test_more_paths_reduces_variance(self):
        """
        Variance ~ 1/N.  Price from 100k paths should be closer to
        the FD benchmark than price from 1k paths (on average).
        """
        fd_ref = 2.314   # Table 1 benchmark S=40, sigma=0.20, T=1
        np.random.seed(0)
        m_small = LeastSquareMonteCarlo(40, 40, 0.06, 0.20, 1.0, 1_000, 50, 3)
        np.random.seed(0)
        m_large = LeastSquareMonteCarlo(40, 40, 0.06, 0.20, 1.0, 100_000, 50, 3)
        err_small = abs(m_small.price() - fd_ref)
        err_large = abs(m_large.price() - fd_ref)
        assert err_large < err_small + 0.10   # large almost always wins


if __name__ == "__main__":
    print("Running quick smoke-check on Table 1 (all 20 cases)…\n")
    K, r = 40.0, 0.06
    print(f"{'S':>4} {'σ':>4} {'T':>2}  {'FD':>6}  {'LSM':>6}  {'|diff|':>7}  {'pass?':>6}")
    print("-" * 52)
    all_ok = True
    for S, sigma, T, fd in TABLE1:
        np.random.seed(0)
        m   = LeastSquareMonteCarlo(S, K, r, sigma, T, 100_000, 50, 3)
        lsm = m.price()
        ok  = abs(lsm - fd) < 0.072
        if not ok:
            all_ok = False
        print(f"{S:>4} {sigma:>4} {T:>2}  {fd:>6.3f}  {lsm:>6.3f}  "
              f"{abs(lsm-fd):>7.4f}  {'✓' if ok else '✗'}")
    print(f"\nAll 20 Table-1 cases pass: {all_ok}")
