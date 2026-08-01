import math
import statistics


def two_sample_z_test_means(
    sample_a: list[float], sample_b: list[float]
) -> tuple[float, float]:
    """
    Two-sample z-test for a difference in means. Appropriate here because both windows have 30+ samples - large enough for the sampling distribution of the mean to be approximately normal (Central Limit Theorem), which is what justifies using a z-test instead of a t-test. For much smaller windows, a t-test would be the more defensible choice.
    """

    mean_a, mean_b = statistics.mean(sample_a), statistics.mean(sample_b)
    var_a, var_b = statistics.variance(sample_a), statistics.variance(sample_b)
    n_a, n_b = len(sample_a), len(sample_b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    z = (mean_a - mean_b) / se
    cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
    p_value = 2 * (1 - cdf)
    return z, p_value
