from scipy import stats
from typing import Union, List, Tuple
import numpy as np


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


def _validate_input(value, name: str, allow_inf: bool = True) -> None:
    arr = np.asarray(value)
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if not allow_inf and np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains Inf values")


class NormalDistribution:
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


class BinomialDistribution:
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


class PoissonDistribution:
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


def create_probability_service() -> DistributionService:
    service = DistributionService()
    service.register_distribution("normal", NormalDistribution)
    service.register_distribution("binomial", BinomialDistribution)
    service.register_distribution("poisson", PoissonDistribution)
    return service
