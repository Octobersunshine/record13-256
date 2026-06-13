from probability_service import create_probability_service
import numpy as np


def main():
    service = create_probability_service()
    np.random.seed(42)

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
    print("【4】分布拟合与假设检验")
    print("=" * 60)

    print("\n--- 4a. 正态分布拟合 (真实 mu=5, sigma=2) ---")
    normal_data = np.random.normal(loc=5, scale=2, size=1000)
    fit_result = service.fit("normal", normal_data)
    print(f"拟合参数: {fit_result['params']}")
    print(f"对数似然: {fit_result['log_likelihood']:.2f}")
    print(f"AIC: {fit_result['aic']:.2f}, BIC: {fit_result['bic']:.2f}")
    gof = service.goodness_of_fit("normal", normal_data)
    print(f"检验方法: {gof['method']}")
    print(f"统计量: {gof['statistic']:.6f}, p值: {gof['p_value']:.6f}")
    print(f"结论: {'接受 H0（正态分布）' if gof['p_value'] > 0.05 else '拒绝 H0'}")

    print("\n--- 4b. 二项分布拟合 (真实 n=20, p=0.3) ---")
    binom_data = np.random.binomial(n=20, p=0.3, size=1000)
    fit_result = service.fit("binomial", binom_data)
    print(f"拟合参数: {fit_result['params']}")
    print(f"对数似然: {fit_result['log_likelihood']:.2f}")
    print(f"AIC: {fit_result['aic']:.2f}, BIC: {fit_result['bic']:.2f}")
    gof = service.goodness_of_fit("binomial", binom_data)
    print(f"检验方法: {gof['method']}")
    print(f"统计量: {gof['statistic']:.6f}, p值: {gof['p_value']:.6f}")
    print(f"结论: {'接受 H0（二项分布）' if gof['p_value'] > 0.05 else '拒绝 H0'}")

    print("\n--- 4c. 泊松分布拟合 (真实 mu=4) ---")
    poisson_data = np.random.poisson(lam=4, size=1000)
    fit_result = service.fit("poisson", poisson_data)
    print(f"拟合参数: {fit_result['params']}")
    print(f"对数似然: {fit_result['log_likelihood']:.2f}")
    print(f"AIC: {fit_result['aic']:.2f}, BIC: {fit_result['bic']:.2f}")
    gof = service.goodness_of_fit("poisson", poisson_data)
    print(f"检验方法: {gof['method']}")
    print(f"统计量: {gof['statistic']:.6f}, p值: {gof['p_value']:.6f}")
    print(f"结论: {'接受 H0（泊松分布）' if gof['p_value'] > 0.05 else '拒绝 H0'}")

    print("\n--- 4d. 自动选择最佳拟合分布 ---")
    auto_result = service.auto_fit(poisson_data, alpha=0.05)
    print(f"样本量: {auto_result['n_observations']}")
    print(f"显著性水平 alpha: {auto_result['alpha']}")
    print("\n候选分布排序（按 p 值降序、AIC 升序）:")
    for i, cand in enumerate(auto_result["candidates"], 1):
        marker = " ★ 最佳" if cand["name"] == auto_result["best_fit"]["name"] else ""
        accept = "✓" if cand["accepted"] else "✗"
        print(f"  {i}. {cand['name']:8s} params={cand['params']}  "
              f"p={cand['p_value']:.4f} AIC={cand['aic']:.1f} [{accept}]{marker}")
    print(f"\n推荐分布: {auto_result['best_fit']['name']}")
    print(f"最佳参数: {auto_result['best_fit']['params']}")
    print(f"p 值: {auto_result['best_fit']['p_value']:.6f}")

    print()
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
