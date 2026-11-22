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

    # 小样本用精确二项；大样本才用卡方近似。
    # 注意：两者必须返回与 method 一致的 p 值，不能标 exact 却给近似值。
    if b_plus_c < 25:
        p_value = _binomial_two_tailed(n10, b_plus_c, 0.5)
        method = "exact"
    else:
        # 卡方(1) 的生存函数：P(X >= chi2) = erfc(sqrt(chi2 / 2))
        p_value = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
        method = "chi2"
    return McNemarResult(
        chi2=round(chi2, 6),
        p_value=round(p_value, 6),
        n10=n10, n01=n01, n11=n11, n00=n00,
        method=method,
    )


def _binomial_two_tailed(k: int, n: int, p: float) -> float:
    """二项分布的双侧精确 p 值。

    在 p=0.5 的对称情形下就是 2 × min(P(X≤k), P(X≥k))，用整数组合数计算，
    不引入正态近似，避免小样本（如 4 vs 0）被算成假显著。
    """
    if n <= 0:
        return 1.0
    k = max(0, min(n, int(k)))

    if abs(p - 0.5) < 1e-12:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)

    # 非 0.5 的一般精确检验：累加概率不大于观测值概率的所有结果
    def _pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    observed = _pmf(k)
    total = 0.0
    for i in range(n + 1):
        pi = _pmf(i)
        if pi <= observed * (1 + 1e-12):
            total += pi
    return min(1.0, max(0.0, total))


def _normal_cdf(z: float) -> float:
    """标准正态 CDF：Φ(z) = 0.5 · erfc(-z / √2)。"""
    return 0.5 * math.erfc(-z / math.sqrt(2))


# --------------------------------------------------------------------------
# Win rate（带 Wilson CI 的成对胜率）
# --------------------------------------------------------------------------

@dataclass
class WinRateResult:
    wins_a: int       # A 胜场数
    wins_b: int       # B 胜场数
    ties: int         # 平局数
    total: int        # 总样本数
    # 全部平局时胜率不可比：point/ci 为 None，禁止用 0.0 伪装成"A 全输"
    point: Optional[float]      # 胜率点估计
    ci_lower: Optional[float]   # Wilson 下界
    ci_upper: Optional[float]   # Wilson 上界
    n_comparable: int # exclude 规则下为 wins_a + wins_b；count_half 下为 total


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
        WinRateResult；exclude 下没有可比值（全平局）时 point/ci 为 None。
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("长度必须相等")
    if tie_rule not in ("exclude", "count_half"):
        raise ValueError(f"未知 tie_rule: {tie_rule}")

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

    if tie_rule == "count_half":
        # 平局各记 0.5：分母为全部样本，点估计与区间都用等价的"半胜"计数
        n_comparable = len(outcomes_a)
        point, lo, hi = wilson_ci(wins_a + 0.5 * ties, n_comparable)
        return WinRateResult(
            wins_a=wins_a, wins_b=wins_b, ties=ties,
            total=len(outcomes_a),
            point=point, ci_lower=lo, ci_upper=hi,
            n_comparable=n_comparable,
        )

    n_comparable = wins_a + wins_b
    if n_comparable == 0:
        # 完全没有胜负可分：胜率不存在，不能返回 0.0
        return WinRateResult(
            wins_a=wins_a, wins_b=wins_b, ties=ties,
            total=len(outcomes_a),
            point=None, ci_lower=None, ci_upper=None,
            n_comparable=0,
        )
    point, lo, hi = wilson_ci(wins_a, n_comparable)
    return WinRateResult(
        wins_a=wins_a, wins_b=wins_b, ties=ties,
        total=len(outcomes_a),
        point=point, ci_lower=lo, ci_upper=hi,
        n_comparable=n_comparable,
    )
