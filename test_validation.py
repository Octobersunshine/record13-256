from probability_service import create_probability_service
import numpy as np


def test_normal_distribution():
    print("=" * 60)
    print("【正态分布】参数校验测试")
    print("=" * 60)

    service = create_probability_service()

    test_cases = [
        ("sigma = 0", {"mu": 0, "sigma": 0}),
        ("sigma = -1", {"mu": 0, "sigma": -1}),
        ("mu = NaN", {"mu": float("nan"), "sigma": 1}),
        ("sigma = NaN", {"mu": 0, "sigma": float("nan")}),
        ("mu = inf", {"mu": float("inf"), "sigma": 1}),
    ]

    for name, params in test_cases:
        try:
            service.pdf("normal", 0, **params)
            print(f"  ❌ {name}: 未抛出异常")
        except ValueError as e:
            print(f"  ✓ {name}: {e}")

    print()
    print("【正态分布】输入值校验测试")
    print("-" * 60)

    dist = service._distributions["normal"](mu=0, sigma=1)

    input_tests = [
        ("pdf(NaN)", lambda: dist.pdf(float("nan"))),
        ("cdf(NaN)", lambda: dist.cdf(float("nan"))),
        ("ppf(NaN)", lambda: dist.ppf(float("nan"))),
        ("ppf(0)", lambda: dist.ppf(0)),
        ("ppf(1)", lambda: dist.ppf(1)),
        ("ppf(-0.1)", lambda: dist.ppf(-0.1)),
        ("ppf(1.1)", lambda: dist.ppf(1.1)),
    ]

    for name, fn in input_tests:
        try:
            result = fn()
            print(f"  ❌ {name}: 返回 {result}, 未抛出异常")
        except ValueError as e:
            print(f"  ✓ {name}: {e}")

    print()


def test_binomial_distribution():
    print("=" * 60)
    print("【二项分布】参数校验测试")
    print("=" * 60)

    service = create_probability_service()

    test_cases = [
        ("n = 0", {"n": 0, "p": 0.5}),
        ("n = -5", {"n": -5, "p": 0.5}),
        ("n = NaN", {"n": float("nan"), "p": 0.5}),
        ("n = 2.5", {"n": 2.5, "p": 0.5}),
        ("p = 0", {"n": 10, "p": 0}),
        ("p = 1", {"n": 10, "p": 1}),
        ("p = -0.1", {"n": 10, "p": -0.1}),
        ("p = 1.1", {"n": 10, "p": 1.1}),
        ("p = NaN", {"n": 10, "p": float("nan")}),
    ]

    for name, params in test_cases:
        try:
            service.pdf("binomial", 5, **params)
            print(f"  ❌ {name}: 未抛出异常")
        except ValueError as e:
            print(f"  ✓ {name}: {e}")

    print()
    print("【二项分布】输入值校验测试")
    print("-" * 60)

    dist = service._distributions["binomial"](n=10, p=0.5)

    input_tests = [
        ("pmf(NaN)", lambda: dist.pmf(float("nan"))),
        ("cdf(NaN)", lambda: dist.cdf(float("nan"))),
        ("ppf(0)", lambda: dist.ppf(0)),
        ("ppf(1)", lambda: dist.ppf(1)),
    ]

    for name, fn in input_tests:
        try:
            result = fn()
            print(f"  ❌ {name}: 返回 {result}, 未抛出异常")
        except ValueError as e:
            print(f"  ✓ {name}: {e}")

    print()


def test_poisson_distribution():
    print("=" * 60)
    print("【泊松分布】参数校验测试")
    print("=" * 60)

    service = create_probability_service()

    test_cases = [
        ("mu = 0", {"mu": 0}),
        ("mu = -1", {"mu": -1}),
        ("mu = NaN", {"mu": float("nan")}),
        ("mu = inf", {"mu": float("inf")}),
    ]

    for name, params in test_cases:
        try:
            service.pdf("poisson", 3, **params)
            print(f"  ❌ {name}: 未抛出异常")
        except ValueError as e:
            print(f"  ✓ {name}: {e}")

    print()
    print("【泊松分布】输入值校验测试")
    print("-" * 60)

    dist = service._distributions["poisson"](mu=3)

    input_tests = [
        ("pmf(NaN)", lambda: dist.pmf(float("nan"))),
        ("cdf(NaN)", lambda: dist.cdf(float("nan"))),
        ("ppf(0)", lambda: dist.ppf(0)),
        ("ppf(1)", lambda: dist.ppf(1)),
    ]

    for name, fn in input_tests:
        try:
            result = fn()
            print(f"  ❌ {name}: 返回 {result}, 未抛出异常")
        except ValueError as e:
            print(f"  ✓ {name}: {e}")

    print()


def test_valid_cases():
    print("=" * 60)
    print("【验证】正常参数仍可正常工作")
    print("=" * 60)

    service = create_probability_service()

    print("\n正态分布 (mu=0, sigma=1):")
    print(f"  pdf(0) = {service.pdf('normal', 0, mu=0, sigma=1):.6f}")
    print(f"  cdf(0) = {service.cdf('normal', 0, mu=0, sigma=1):.6f}")
    print(f"  ppf(0.975) = {service.ppf('normal', 0.975, mu=0, sigma=1):.6f}")
    print(f"  mean = {service.mean('normal', mu=0, sigma=1)}")

    print("\n二项分布 (n=10, p=0.5):")
    print(f"  pmf(5) = {service.pdf('binomial', 5, n=10, p=0.5):.6f}")
    print(f"  cdf(5) = {service.cdf('binomial', 5, n=10, p=0.5):.6f}")
    print(f"  ppf(0.5) = {service.ppf('binomial', 0.5, n=10, p=0.5)}")
    print(f"  mean = {service.mean('binomial', n=10, p=0.5)}")

    print("\n泊松分布 (mu=3):")
    print(f"  pmf(3) = {service.pdf('poisson', 3, mu=3):.6f}")
    print(f"  cdf(3) = {service.cdf('poisson', 3, mu=3):.6f}")
    print(f"  ppf(0.5) = {service.ppf('poisson', 0.5, mu=3)}")
    print(f"  mean = {service.mean('poisson', mu=3)}")

    print("\n数组输入测试:")
    x_arr = np.array([-1, 0, 1])
    pdf_arr = service.pdf("normal", x_arr, mu=0, sigma=1)
    print(f"  normal.pdf({x_arr.tolist()}) = {pdf_arr.round(6).tolist()}")

    print()


if __name__ == "__main__":
    test_normal_distribution()
    test_binomial_distribution()
    test_poisson_distribution()
    test_valid_cases()
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
