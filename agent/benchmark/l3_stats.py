"""L3 统计层：跨臂对比的胜率、置信区间、显著性检验

为什么独立模块：
- 不依赖 scipy / statsmodels：Wilson CI 和 McNemar 公式都不复杂，
  手写 = 零额外依赖 + 可审计（面试里也讲得清公式）
- 纯函数：输入是数值，输出是数值，可独立单测

公式参考：
- Wilson 95% CI：Agresti & Coull (1998) 推荐的二项比例区间
- McNemar：配对二分类数据的同质性检验；小样本用精确二项检验替代
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Wilson score interval（二项比例的 95% 置信区间）
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """计算 Wilson 区间。

    Args:
        k: 成功数（events）
        n: 总试验数（trials），必须 > 0
        confidence: 置信水平，默认 0.95

    Returns:
        (point, lower, upper)，所有值在 [0, 1] 范围；n=0 时返回 (0.0, 0.0, 0.0)

    参考：Agresti & Coull, "Approximate is Better than 'Exact' for Interval Estimation
    of Binomial Proportions", American Statistician, 1998
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    if k < 0:
        k = 0
    if k > n:
        k = n

    z = _z_for_confidence(confidence)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(p, 6), round(lower, 6), round(upper, 6)


def _z_for_confidence(confidence: float) -> float:
    """标准正态分布的分位数（反 erf）。精确到 1e-4。

    常见值：0.90→1.6449, 0.95→1.9600, 0.99→2.5758
    用 Abramowitz & Stegun 7.1.26 近似公式。
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f"confidence={confidence} 必须在 (0,1) 区间")
    # 双侧：置信水平为 confidence 的 Z 值 = 反函数(1 - (1-c)/2)
    # _inverse_normal_cdf 返回的是分位点，需要取绝对值
    tail = 1 - (1 - confidence) / 2
    return abs(_inverse_normal_cdf(tail))


def _inverse_normal_cdf(p: float) -> float:
    """Rational approximation to the inverse of the standard normal CDF.

    Accuracy: absolute error < 1.15e-9 for |p - 0.5| < 0.49
    Source: Peter Acklam, https://web.archive.org/web/20151030215612/https://people.math.sfu.ca/~cbm/aands/page_253.htm
    """
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0

    # Coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    # p > phigh
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


# --------------------------------------------------------------------------
# McNemar's test（配对二分类的同质性检验）
# --------------------------------------------------------------------------

@dataclass
class McNemarResult:
    chi2: float       # 卡方统计量（连续性校正后）
    p_value: float    # 精确二项检验 p 值（更准确）
    n10: int          # A=fail, B=success 的数量
    n01: int          # A=success, B=fail 的数量
    n11: int          # 两者都 success
    n00: int          # 两者都 fail
    method: str       # "chi2" 或 "exact"


def mcnemar(outcomes_a: List[bool], outcomes_b: List[bool]) -> McNemarResult:
    """McNemar's test for paired binary data.

    Args:
        outcomes_a: arm A 的每样本 outcome（True = success/blocked, False = fail）
        outcomes_b: arm B 的每样本 outcome，长度必须与 outcomes_a 一致

    Returns:
        McNemarResult

    小样本（b+c < 25）用精确二项检验；大样本用连续性校正的卡方。
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(f"长度不等: A={len(outcomes_a)}, B={len(outcomes_b)}")
    n = len(outcomes_a)

    n11 = n10 = n01 = n00 = 0
    for a, b in zip(outcomes_a, outcomes_b):
        a_b = bool(a)
        b_b = bool(b)
        if a_b and b_b:
            n11 += 1
        elif a_b and not b_b:
            n10 += 1
        elif not a_b and b_b:
            n01 += 1
        else:
            n00 += 1

    b_plus_c = n10 + n01
    if b_plus_c == 0:
        # 完全一致，没有可检验的差异
        return McNemarResult(chi2=0.0, p_value=1.0,
                             n10=n10, n01=n01, n11=n11, n00=n00,
                             method="exact")

    # 连续性校正的卡方
    chi2 = (abs(n10 - n01) - 1) ** 2 / b_plus_c
    # 精确二项检验：用 binomial 分布检验 n10 是否偏离 b_plus_c 的 0.5
    p_value = _binomial_two_tailed(n10, b_plus_c, 0.5)

    method = "exact" if b_plus_c < 25 else "chi2"
    return McNemarResult(
        chi2=round(chi2, 6),
        p_value=round(p_value, 6),
        n10=n10, n01=n01, n11=n11, n00=n00,
        method=method,
    )


def _binomial_two_tailed(k: int, n: int, p: float) -> float:
    """二项分布的双侧 p 值。"""
    if n <= 0:
        return 1.0

    # 用正态近似的二项 p 值（在 p=0.5 对称时双侧近似 = 单侧 × 2）
    mu = n * p
    var = n * p * (1 - p)
    sigma = math.sqrt(var)

    if sigma < 1e-10:
        return 1.0 if k == int(mu) else 0.0

    # 连续性校正：P(X >= k) ≈ P(Z >= (k - 0.5 - mu) / sigma)
    z_upper = (k - 0.5 - mu) / sigma
    # P(X <= k-1) ≈ P(Z <= (k - 0.5 - mu) / sigma)（用同一连续性校正点）
    # 但双侧 = 2 * min(tail(k >= mu), tail(k <= mu))

    cdf_k = _normal_cdf(z_upper)   # P(Z <= z_upper)
    if k >= mu:
        tail_upper = 1.0 - cdf_k    # P(Z >= z_upper)
        tail_lower = _normal_cdf(-z_upper)  # 对称下界
    else:
        tail_lower = cdf_k           # P(Z <= z_upper)
        tail_upper = 1.0 - cdf_k

    p_two = 2 * min(tail_lower, tail_upper)
    return min(1.0, max(0.0, p_two))


def _normal_cdf(z: float) -> float:
    """标准正态 CDF 的数值积分近似（Abramowitz & Stegun 7.1.26）。"""
    # 用 erf 近似：erf(z) ≈ 1 - (a1*t + a2*t² + ... + a5*t⁵)*exp(-z²)
    z_abs = abs(z)
    t = 1.0 / (1.0 + 0.3275911 * z_abs)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    erf_approx = 1.0 - ((a1 * t + a2 * t * t + a3 * t ** 3 +
                         a4 * t ** 4 + a5 * t ** 5) * math.exp(-z_abs * z_abs))
    half = 0.5 * (1.0 + erf_approx)
    return half if z >= 0 else (1.0 - half)


# --------------------------------------------------------------------------
# Win rate（带 Wilson CI 的成对胜率）
# --------------------------------------------------------------------------

@dataclass
class WinRateResult:
    wins_a: int       # A 胜场数
    wins_b: int       # B 胜场数
    ties: int         # 平局数
    total: int        # 总样本数
    point: float      # 胜率点估计 = wins_a / (wins_a + wins_b)
    ci_lower: float   # Wilson 下界
    ci_upper: float   # Wilson 上界
    n_comparable: int # wins_a + wins_b（排除 ties 的分母）


def win_rate(
    outcomes_a: List[bool],
    outcomes_b: List[bool],
    *,
    tie_rule: str = "exclude",
) -> WinRateResult:
    """计算臂 A 对臂 B 的成对胜率。

    Args:
        outcomes_a: arm A 每样本 outcome（True=success/blocked）
        outcomes_b: arm B 每样本 outcome
        tie_rule: 平局处理方式 —— "exclude"（排除出分母，默认）或 "count_half"（各计 0.5）

    Returns:
        WinRateResult
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("长度必须相等")

    wins_a = wins_b = ties = 0
    for a, b in zip(outcomes_a, outcomes_b):
        a_b = bool(a)
        b_b = bool(b)
        if a_b and not b_b:
            wins_a += 1
        elif b_b and not a_b:
            wins_b += 1
        else:
            ties += 1

    n_comparable = wins_a + wins_b
    point, lo, hi = wilson_ci(wins_a, n_comparable)

    return WinRateResult(
        wins_a=wins_a, wins_b=wins_b, ties=ties,
        total=len(outcomes_a),
        point=point, ci_lower=lo, ci_upper=hi,
        n_comparable=n_comparable,
    )
