from scipy import stats
from typing import Union, List, Dict, Any, Optional
import numpy as np


def _validate_input(value, name: str, allow_inf: bool = True) -> None:
    arr = np.asarray(value)
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if not allow_inf and np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains Inf values")


def _validate_data(data, name: str = "data") -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if len(arr) < 2:
        raise ValueError(f"{name} must contain at least 2 observations")
    _validate_input(arr, name, allow_inf=False)
    return arr


def _chi_square_test_discrete(
    data: np.ndarray,
    pmf_func,
    n_params: int,
    min_expected: float = 5.0,
) -> Dict[str, float]:
    data = data.astype(int)
    n = len(data)
    data_min, data_max = int(data.min()), int(data.max())

    observed = []
    expected = []
    current_obs = 0
    current_exp = 0

    for k in range(data_min, data_max + 1):
        obs_k = int(np.sum(data == k))
        exp_k = pmf_func(k) * n
        current_obs += obs_k
        current_exp += exp_k
        if current_exp >= min_expected:
            observed.append(current_obs)
            expected.append(current_exp)
            current_obs = 0
            current_exp = 0

    if current_obs > 0 or current_exp > 0:
        if len(expected) > 0 and current_exp < min_expected:
            observed[-1] += current_obs
            expected[-1] += current_exp
        else:
            observed.append(current_obs)
            expected.append(current_exp)

    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    expected = expected * (observed.sum() / expected.sum())

    ddof = n_params
    dof = max(len(observed) - 1 - ddof, 1)
    chi2_stat, p_value = stats.chisquare(observed, expected, ddof=ddof)
    return {
        "statistic": float(chi2_stat),
        "p_value": float(p_value),
        "dof": dof,
        "method": "Chi-square test",
    }


class DistributionService:
    def __init__(self):
        self._distributions = {}

    def register_distribution(self, name: str, distribution) -> None:
        self._distributions[name] = distribution

    def list_distributions(self) -> List[str]:
        return list(self._distributions.keys())

    def pdf(self, dist_name: str, x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return dist.pdf(x)

    def cdf(self, dist_name: str, x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return dist.cdf(x)

    def ppf(self, dist_name: str, q: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return dist.ppf(q)

    def mean(self, dist_name: str, **kwargs) -> float:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return dist.mean()

    def var(self, dist_name: str, **kwargs) -> float:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return dist.var()

    def std(self, dist_name: str, **kwargs) -> float:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return dist.std()

    def summary(self, dist_name: str, **kwargs) -> dict:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist = self._distributions[dist_name](**kwargs)
        return {
            "name": dist_name,
            "params": kwargs,
            "mean": float(dist.mean()),
            "variance": float(dist.var()),
            "std": float(dist.std()),
        }

    def fit(self, dist_name: str, data, **kwargs) -> Dict[str, Any]:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist_cls = self._distributions[dist_name]
        if not hasattr(dist_cls, "fit"):
            raise NotImplementedError(f"Distribution '{dist_name}' does not support fitting")
        return dist_cls.fit(data, **kwargs)

    def goodness_of_fit(self, dist_name: str, data, **kwargs) -> Dict[str, Any]:
        if dist_name not in self._distributions:
            raise ValueError(f"Distribution '{dist_name}' not registered")
        dist_cls = self._distributions[dist_name]
        if not hasattr(dist_cls, "goodness_of_fit"):
            raise NotImplementedError(f"Distribution '{dist_name}' does not support goodness_of_fit")
        return dist_cls.goodness_of_fit(data, **kwargs)

    def auto_fit(
        self,
        data,
        candidates: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        if candidates is None:
            candidates = self.list_distributions()
        else:
            for name in candidates:
                if name not in self._distributions:
                    raise ValueError(f"Distribution '{name}' not registered")

        results = []
        data_arr = _validate_data(data)

        for name in candidates:
            dist_cls = self._distributions[name]
            if not hasattr(dist_cls, "fit") or not hasattr(dist_cls, "goodness_of_fit"):
                continue
            try:
                fit_result = dist_cls.fit(data_arr)
                gof_result = dist_cls.goodness_of_fit(data_arr)
                results.append({
                    "name": name,
                    "params": fit_result["params"],
                    "log_likelihood": fit_result["log_likelihood"],
                    "aic": fit_result["aic"],
                    "bic": fit_result["bic"],
                    "test_statistic": gof_result["statistic"],
                    "p_value": gof_result["p_value"],
                    "test_method": gof_result["method"],
                    "accepted": gof_result["p_value"] > alpha,
                })
            except Exception:
                continue

        if not results:
            raise ValueError("No distribution could be fitted to the data")

        results.sort(key=lambda r: (-r["p_value"], r["aic"]))
        best = results[0]

        return {
            "best_fit": best,
            "candidates": results,
            "alpha": alpha,
            "n_observations": len(data_arr),
        }


class NormalDistribution:
    is_discrete = False
    n_params = 2

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if np.isnan(mu) or np.isinf(mu):
            raise ValueError("mu must be a finite number")
        if np.isnan(sigma) or np.isinf(sigma):
            raise ValueError("sigma must be a finite number")
        if sigma <= 0:
            raise ValueError("sigma must be positive (sigma > 0)")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(x, "x")
        return self._dist.pdf(x)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(x, "x")
        return self._dist.cdf(x)

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(q, "q")
        if np.any(q <= 0) or np.any(q >= 1):
            raise ValueError("q must be in (0, 1) exclusive")
        return self._dist.ppf(q)

    def mean(self) -> float:
        return self._dist.mean()

    def var(self) -> float:
        return self._dist.var()

    def std(self) -> float:
        return self._dist.std()

    @staticmethod
    def fit(data) -> Dict[str, Any]:
        data_arr = _validate_data(data)
        mu_hat, sigma_hat = stats.norm.fit(data_arr)
        fitted = NormalDistribution(mu=mu_hat, sigma=sigma_hat)
        log_likelihood = float(np.sum(fitted._dist.logpdf(data_arr)))
        n = len(data_arr)
        k = NormalDistribution.n_params
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        return {
            "params": {"mu": float(mu_hat), "sigma": float(sigma_hat)},
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "bic": float(bic),
            "n_observations": n,
        }

    @staticmethod
    def goodness_of_fit(data, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        data_arr = _validate_data(data)
        if params is None:
            fit_result = NormalDistribution.fit(data_arr)
            params = fit_result["params"]
        mu = params["mu"]
        sigma = params["sigma"]
        stat, p_value = stats.kstest(data_arr, "norm", args=(mu, sigma))
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "method": "Kolmogorov-Smirnov test",
            "params": {"mu": mu, "sigma": sigma},
        }


class BinomialDistribution:
    is_discrete = True
    n_params = 2

    def __init__(self, n: int, p: float):
        if np.isnan(n) or np.isinf(n):
            raise ValueError("n must be a finite number")
        if not float(n).is_integer():
            raise ValueError("n must be an integer")
        if n <= 0:
            raise ValueError("n must be positive (n > 0)")
        if np.isnan(p) or np.isinf(p):
            raise ValueError("p must be a finite number")
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1) exclusive")
        self.n = int(n)
        self.p = p
        self._dist = stats.binom(n=self.n, p=p)

    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(k, "k")
        return self._dist.pmf(k)

    def pdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.pmf(k)

    def cdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(k, "k")
        return self._dist.cdf(k)

    def ppf(self, q: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        _validate_input(q, "q")
        if np.any(q <= 0) or np.any(q >= 1):
            raise ValueError("q must be in (0, 1) exclusive")
        return self._dist.ppf(q)

    def mean(self) -> float:
        return self._dist.mean()

    def var(self) -> float:
        return self._dist.var()

    def std(self) -> float:
        return self._dist.std()

    @staticmethod
    def fit(data) -> Dict[str, Any]:
        data_arr = _validate_data(data)
        if np.any(data_arr < 0) or np.any(data_arr != np.floor(data_arr)):
            raise ValueError("Binomial data must be non-negative integers")

        mean_x = float(np.mean(data_arr))
        var_x = float(np.var(data_arr, ddof=0))

        if var_x >= mean_x or mean_x <= 0:
            n_start = max(int(np.max(data_arr)) + 1, 2)
        else:
            p_mom = 1 - var_x / mean_x
            n_mom = int(round(mean_x / p_mom))
            n_start = max(n_mom, int(np.max(data_arr)) + 1)

        best_n, best_p, best_ll = None, None, -np.inf
        search_range = range(max(2, int(np.max(data_arr))), n_start + 30)

        for n_candidate in search_range:
            p_candidate = mean_x / n_candidate
            if not (0 < p_candidate < 1):
                continue
            try:
                ll = float(np.sum(stats.binom.logpmf(data_arr.astype(int), n=n_candidate, p=p_candidate)))
                if np.isfinite(ll) and ll > best_ll:
                    best_ll = ll
                    best_n = n_candidate
                    best_p = p_candidate
            except Exception:
                continue

        if best_n is None:
            raise ValueError("Failed to fit binomial distribution")

        n = len(data_arr)
        k = BinomialDistribution.n_params
        aic = 2 * k - 2 * best_ll
        bic = k * np.log(n) - 2 * best_ll

        return {
            "params": {"n": int(best_n), "p": float(best_p)},
            "log_likelihood": float(best_ll),
            "aic": float(aic),
            "bic": float(bic),
            "n_observations": n,
        }

    @staticmethod
    def goodness_of_fit(data, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data_arr = _validate_data(data)
        if params is None:
            fit_result = BinomialDistribution.fit(data_arr)
            params = fit_result["params"]
        n = params["n"]
        p = params["p"]
        pmf_func = lambda k: stats.binom.pmf(k, n=n, p=p)
        result = _chi_square_test_discrete(data_arr, pmf_func, n_params=BinomialDistribution.n_params)
        result["params"] = {"n": n, "p": p}
        return result


class PoissonDistribution:
    is_discrete = True
    n_params = 1

    def __init__(self, mu: float):
        if np.isnan(mu) or np.isinf(mu):
            raise ValueError("mu must be a finite number")
        if mu <= 0:
            raise ValueError("mu must be positive (mu > 0)")
        self.mu = mu
        self._dist = stats.poisson(mu=mu)

    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(k, "k")
        return self._dist.pmf(k)

    def pdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return self.pmf(k)

    def cdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        _validate_input(k, "k")
        return self._dist.cdf(k)

    def ppf(self, q: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        _validate_input(q, "q")
        if np.any(q <= 0) or np.any(q >= 1):
            raise ValueError("q must be in (0, 1) exclusive")
        return self._dist.ppf(q)

    def mean(self) -> float:
        return self._dist.mean()

    def var(self) -> float:
        return self._dist.var()

    def std(self) -> float:
        return self._dist.std()

    @staticmethod
    def fit(data) -> Dict[str, Any]:
        data_arr = _validate_data(data)
        if np.any(data_arr < 0) or np.any(data_arr != np.floor(data_arr)):
            raise ValueError("Poisson data must be non-negative integers")

        mu_hat = float(np.mean(data_arr))
        if mu_hat <= 0:
            raise ValueError("Cannot fit Poisson distribution: data mean is non-positive")

        fitted = PoissonDistribution(mu=mu_hat)
        log_likelihood = float(np.sum(fitted._dist.logpmf(data_arr.astype(int))))
        n = len(data_arr)
        k = PoissonDistribution.n_params
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return {
            "params": {"mu": mu_hat},
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "bic": float(bic),
            "n_observations": n,
        }

    @staticmethod
    def goodness_of_fit(data, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        data_arr = _validate_data(data)
        if params is None:
            fit_result = PoissonDistribution.fit(data_arr)
            params = fit_result["params"]
        mu = params["mu"]
        pmf_func = lambda k: stats.poisson.pmf(k, mu=mu)
        result = _chi_square_test_discrete(data_arr, pmf_func, n_params=PoissonDistribution.n_params)
        result["params"] = {"mu": mu}
        return result


def create_probability_service() -> DistributionService:
    service = DistributionService()
    service.register_distribution("normal", NormalDistribution)
    service.register_distribution("binomial", BinomialDistribution)
    service.register_distribution("poisson", PoissonDistribution)
    return service
