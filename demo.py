from probability_service import create_probability_service
import numpy as np


def main():
    service = create_probability_service()

    print("=" * 60)
    print("概率分布服务 - 支持分布列表")
    print("=" * 60)
    print(service.list_distributions())
    print()

    print("=" * 60)
    print("【1】正态分布示例 (mu=0, sigma=1)")
    print("=" * 60)
    normal_summary = service.summary("normal", mu=0, sigma=1)
    print(f"统计摘要: {normal_summary}")

    x_vals = np.array([-2, -1, 0, 1, 2])
    pdf_vals = service.pdf("normal", x_vals, mu=0, sigma=1)
    cdf_vals = service.cdf("normal", x_vals, mu=0, sigma=1)
    print(f"\nx = {x_vals}")
    print(f"PDF = {pdf_vals}")
    print(f"CDF = {cdf_vals}")

    q_vals = np.array([0.025, 0.5, 0.975])
    ppf_vals = service.ppf("normal", q_vals, mu=0, sigma=1)
    print(f"\n分位数 q = {q_vals}")
    print(f"PPF = {ppf_vals}")
    print()

    print("=" * 60)
    print("【2】二项分布示例 (n=10, p=0.5)")
    print("=" * 60)
    binom_summary = service.summary("binomial", n=10, p=0.5)
    print(f"统计摘要: {binom_summary}")

    k_vals = np.array([0, 3, 5, 7, 10])
    pmf_vals = service.pdf("binomial", k_vals, n=10, p=0.5)
    cdf_vals = service.cdf("binomial", k_vals, n=10, p=0.5)
    print(f"\nk = {k_vals}")
    print(f"PMF = {pmf_vals}")
    print(f"CDF = {cdf_vals}")

    q_vals = np.array([0.05, 0.5, 0.95])
    ppf_vals = service.ppf("binomial", q_vals, n=10, p=0.5)
    print(f"\n分位数 q = {q_vals}")
    print(f"PPF = {ppf_vals}")
    print()

    print("=" * 60)
    print("【3】泊松分布示例 (mu=3)")
    print("=" * 60)
    poisson_summary = service.summary("poisson", mu=3)
    print(f"统计摘要: {poisson_summary}")

    k_vals = np.array([0, 2, 3, 5, 10])
    pmf_vals = service.pdf("poisson", k_vals, mu=3)
    cdf_vals = service.cdf("poisson", k_vals, mu=3)
    print(f"\nk = {k_vals}")
    print(f"PMF = {pmf_vals}")
    print(f"CDF = {cdf_vals}")

    q_vals = np.array([0.1, 0.5, 0.9])
    ppf_vals = service.ppf("poisson", q_vals, mu=3)
    print(f"\n分位数 q = {q_vals}")
    print(f"PPF = {ppf_vals}")
    print()

    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
