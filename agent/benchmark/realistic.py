"""按私有轨聚合分布生成公开轨数据（中立域）

约束：
- 只读取 `profiles.py` 中的聚合参数，不读取私有文件
- 只复用本模块内的通用售后话术，不复制任何私有原文或业务标签
- 同 seed 必然产出同一份数据
"""

import random
import re
from typing import Dict, List, Optional, Sequence, Set

from .profiles import (
    REALISTIC_ASSISTANT_UNIQUE_RATE,
    REALISTIC_INTERNAL_ASSISTANT_DUP_RATE,
    REALISTIC_TURN_HISTOGRAM,
)

SCENARIOS = ["S1_咨询", "S2_催办", "S3_退换", "S4_投诉", "S5_回访"]
PRODUCTS = ["示例产品甲", "示例产品乙", "示例产品丙"]

SYSTEM_TEMPLATE = """# 角色
售后回访专员

## 案例信息
回访目标：核实客户对本次售后处理结果的确认情况
所属机构：示例服务中心（华东）
业务类型：订单售后回访
专员工号：A-{staff}
案件编号：{case_id}

## 标签信息
场景标签：{scenario}
产品名称：{product}
订单金额：{amount}
联系时间：2026-11-12 10:00:00

## 服务规范
先确认客户身份与订单信息；围绕客户当前诉求给出明确处理方案；
不承诺系统外时效，不虚构金额与编号；通话结束前复述处理结论并确认客户是否听清。

## 案例摘要
客户在本平台购买示例产品后反馈物流进度、包装状态或换货流程问题，
本次回访需要核实问题是否仍然存在，并记录客户对处理结果的接受程度。
若客户提出新的诉求，应先记录再转交对应同事，不在本次通话中承诺额外补偿。
"""

# ---- assistant 话术池：16 × 16 × 16 = 4096 个三段组合，四段组合再乘 16 ----
ASSISTANT_OPENINGS = [
    "您好，我是售后回访专员。",
    "感谢您接听本次回访电话。",
    "我这边看到您提交了售后反馈。",
    "关于您反馈的问题，我已经记录。",
    "先和您确认一下当前的处理进度。",
    "您反馈的情况我这边已经了解。",
    "这边帮您核对一下订单信息。",
    "我先把您说的问题复述一遍。",
    "感谢您补充这些细节。",
    "理解您的心情，我会如实记录。",
    "为了准确处理，我先确认几个信息。",
    "您提到的这个情况我这边能看到。",
    "本次回访主要是确认处理结果。",
    "我这边同步一下当前的进展。",
    "您方便的话，我占用您几分钟。",
    "关于这个订单，我这边有一条新进展。",
]
ASSISTANT_ACTIONS = [
    "订单目前处于分拣环节，预计两个工作日内更新物流。",
    "换货申请已经登记，仓库收到退回件后会重新发出。",
    "您反馈的包装破损情况已转交仓储同事核实。",
    "这边会帮您催办，有进展会通过短信通知。",
    "退款会按原支付路径退回，到账时间以银行处理为准。",
    "我已经把您的要求备注到工单里。",
    "对应同事会在今天下班前给您回电确认。",
    "当前没有额外费用，订单信息以系统记录为准。",
    "如果需要修改收货地址，我这边可以协助登记。",
    "物流信息暂时没有更新，我会继续跟进。",
    "您可以在订单详情页查看最新处理状态。",
    "这个型号的库存情况我需要再核实一下。",
    "我这边先按换货流程为您提交申请。",
    "补偿方案需要对应同事审核后才能确认。",
    "我已经记录了您希望尽快处理的要求。",
    "如果超过时效仍未更新，可以再次联系我们。",
]
ASSISTANT_DETAILS = [
    "目前系统显示的处理节点和您看到的一致。",
    "仓库会在收到退回件后重新检查商品状态。",
    "物流更新可能存在一定延迟，请以实际到货为准。",
    "退款到账前请保留好支付凭证。",
    "换货商品需要保持包装和配件完整。",
    "如果地址有变更，请在我这边登记后等待确认。",
    "工单已经进入对应同事的处理队列。",
    "短信通知会发送到您下单时预留的手机号。",
    "本次处理不会产生额外服务费用。",
    "您可以随时在订单页面查看处理记录。",
    "如果商品存在质量问题，我们会按流程核实。",
    "加急申请需要根据当前库存情况确认。",
    "补偿方案以审核结果为准，我这边不能提前承诺。",
    "通话记录会保留，方便后续核对处理过程。",
    "如果今天没有收到通知，可以再次联系客服。",
    "您反馈的照片已经附在对应的工单里。",
]
ASSISTANT_CLOSINGS = [
    "您看这样处理可以吗？",
    "还有其他需要我这边记录的吗？",
    "稍后请留意短信通知。",
    "感谢您的理解与配合。",
    "如有变化我会及时联系您。",
    "请您保持电话畅通。",
    "我这边就先不打扰您了。",
    "后续有结果会第一时间同步。",
    "您先别着急，我们按流程处理。",
    "如果方便，请您确认一下以上信息。",
    "本次回访到这里，感谢您接听。",
    "您反馈的问题我会持续跟进。",
    "请以系统显示的状态为准。",
    "有新的进展我再和您联系。",
    "您看还有哪里需要补充？",
    "处理完成后会有结果通知。",
]

# ---- user 话术池：8 × 8 × 24 × 8 = 12288 个四段组合 ----
USER_TIMES = ["上周", "昨天", "今天", "前两天", "刚刚", "这周", "上个月", "最近"]
USER_ACTIONS = [
    "下的单", "买的商品", "申请的换货", "反馈的问题",
    "收到的包裹", "联系的客服", "看到的物流", "填的地址",
]
USER_TOPICS = [
    "还没收到", "物流没更新", "想催一下", "型号不一样", "包装有破损", "想换货",
    "退款没到账", "想改地址", "想知道进度", "需要寄回吗", "短信准吗", "想转人工",
    "想核对金额", "要原包装吗", "能取消吗", "能加急吗", "补偿算数吗", "几点下班",
    "还会回访吗", "该找谁", "不会操作", "看不懂页面", "能发短信吗", "今天能处理吗",
]
USER_ACKS = ["谢谢", "麻烦尽快", "我等着", "可以", "好的", "辛苦了", "那就这样", "我再看看"]

_ID_PATTERN = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")

ASSISTANT_REUSE_RATE = 1 - REALISTIC_ASSISTANT_UNIQUE_RATE
# 首轮必须新建（带案件号，供 D2 注入使用），会额外贡献约 9% 的唯一 user 消息；
# 非首轮复用率相应上调，使整体唯一率仍落在私有轨 28.5% 附近。
USER_REUSE_RATE = 0.79


def _compose(pools: Sequence[Sequence[str]], index: int) -> str:
    """按混合进制把 index 解码成各池的一个元素；index 超出组合数时回绕。"""
    picked: List[str] = []
    for pool in reversed(pools):
        index, j = divmod(index, len(pool))
        picked.append(pool[j])
    return "".join(reversed(picked))


def _similar(a: str, b: str) -> bool:
    """字符集合 Jaccard ≥ 0.8 视为高度相似；与私有轨相邻轮统计口径一致。"""
    if not a or not b:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb)) >= 0.8


def _user_text(index: int, with_ack: bool) -> str:
    if with_ack:
        index, ack_i = divmod(index, len(USER_ACKS))
    else:
        ack_i = None
    index, topic_i = divmod(index, len(USER_TOPICS))
    index, action_i = divmod(index, len(USER_ACTIONS))
    time = USER_TIMES[index % len(USER_TIMES)]
    head = f"{time}{USER_ACTIONS[action_i]}，{USER_TOPICS[topic_i]}"
    return f"{head}，{USER_ACKS[ack_i]}。" if with_ack else f"{head}。"


def allocate_turns(n: int) -> List[int]:
    """按私有轨轮次直方图分配 n 条对话的轮数，保证总和精确等于 n。"""
    if n < 1:
        raise ValueError("n 必须为正整数")
    total_weight = sum(REALISTIC_TURN_HISTOGRAM.values())
    rows = []
    assigned = 0
    for turns, weight in sorted(REALISTIC_TURN_HISTOGRAM.items()):
        quota = n * weight / total_weight
        base = int(quota)
        rows.append([turns, base, quota - base])
        assigned += base
    remaining = n - assigned
    rows.sort(key=lambda r: (-r[2], r[0]))
    for i in range(remaining):
        rows[i][1] += 1
    out: List[int] = []
    for turns, count, _ in rows:
        out.extend([turns] * count)
    return out


class RealisticState:
    """跨对话共享的话术池与计数器，用于控制全局重复率。"""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.user_pool: List[str] = []
        self.assistant_pool: List[str] = []
        self.user_three_counter = 0
        self.user_four_counter = 0
        self.assistant_three_counter = 0
        self.assistant_four_counter = 0

    def _reuse(self, pool: List[str], seen: Set[str],
               previous: Optional[str] = None) -> Optional[str]:
        if not pool:
            return None
        for _ in range(8):
            text = pool[self.rng.randrange(len(pool))]
            if text in seen or _ID_PATTERN.search(text):
                continue
            if previous and _similar(text, previous):
                continue
            return text
        return None

    def next_user(self, seen: Set[str], case_id: Optional[str] = None) -> str:
        # 首轮要带案件号，不能复用不含编号的通用话术
        if case_id is None and self.rng.random() < USER_REUSE_RATE:
            text = self._reuse(self.user_pool, seen)
            if text is not None:
                seen.add(text)
                return text
        for _ in range(16):
            if self.rng.random() < 0.5:
                text = _user_text(self.user_four_counter, True)
                self.user_four_counter += 1
            else:
                text = _user_text(self.user_three_counter, False)
                self.user_three_counter += 1
            if text not in seen:
                break
        if case_id:
            text = f"{text.rstrip('。')}（案件号 {case_id}）。"
        else:
            self.user_pool.append(text)
        seen.add(text)
        return text

    def next_assistant(self, seen: Set[str],
                       previous: Optional[str] = None) -> str:
        if self.rng.random() < ASSISTANT_REUSE_RATE:
            text = self._reuse(self.assistant_pool, seen, previous)
            if text is not None:
                seen.add(text)
                return text
        for _ in range(16):
            if self.rng.random() < 0.4:
                text = _compose(
                    [ASSISTANT_OPENINGS, ASSISTANT_ACTIONS, ASSISTANT_DETAILS,
                     ASSISTANT_CLOSINGS],
                    self.assistant_four_counter,
                )
                self.assistant_four_counter += 1
            else:
                text = _compose(
                    [ASSISTANT_OPENINGS, ASSISTANT_ACTIONS, ASSISTANT_CLOSINGS],
                    self.assistant_three_counter,
                )
                self.assistant_three_counter += 1
            if text not in seen and not (previous and _similar(text, previous)):
                break
        self.assistant_pool.append(text)
        seen.add(text)
        return text


def _duplicate_assistant(messages: List[dict], rng: random.Random) -> None:
    """在单条对话内部复制一条非相邻 assistant 消息，复现私有轨 3.3% 的重叠。"""
    a_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    pairs = [
        (a_idx[p], a_idx[q])
        for p in range(len(a_idx)) for q in range(p + 1, len(a_idx))
        if q - p > 1
        and messages[a_idx[p]].get("content") != messages[a_idx[q]].get("content")
    ]
    if not pairs:
        return
    src, dst = pairs[rng.randrange(len(pairs))]
    messages[dst] = {**messages[dst], "content": messages[src]["content"]}


def make_dialogue(idx: int, n_turns: int, state: RealisticState) -> dict:
    rng = state.rng
    scenario = SCENARIOS[idx % len(SCENARIOS)]
    product = PRODUCTS[idx % len(PRODUCTS)]
    case_id = f"CASE-{rng.randint(100000, 999999)}"
    amount = f"{rng.uniform(500, 20000):.2f}"
    system = SYSTEM_TEMPLATE.format(
        staff=rng.randint(1000, 9999),
        case_id=case_id,
        scenario=scenario,
        product=product,
        amount=amount,
    )
    messages = [{"role": "system", "content": system}]
    user_seen: Set[str] = set()
    assistant_seen: Set[str] = set()
    previous_assistant = None
    for t in range(n_turns):
        messages.append({"role": "user", "content": state.next_user(
            user_seen, case_id if t == 0 else None)})
        content = state.next_assistant(assistant_seen, previous_assistant)
        messages.append({"role": "assistant", "content": content, "loss": "True"})
        previous_assistant = content
    if rng.random() < REALISTIC_INTERNAL_ASSISTANT_DUP_RATE:
        _duplicate_assistant(messages, rng)
    return {"messages": messages}


def generate_dialogues(n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    turns = allocate_turns(n)
    rng.shuffle(turns)
    state = RealisticState(rng)
    return [make_dialogue(i, turns[i], state) for i in range(n)]


def fixture_stats(dialogues: Sequence[dict]) -> Dict[str, float]:
    """计算校准指标，供测试与作品集核验使用；口径与私有轨统计一致。"""
    a_all, u_all = [], []
    internal_dup = 0
    for d in dialogues:
        msgs = d.get("messages") or []
        a = [_digit_norm(m.get("content")) for m in msgs if m.get("role") == "assistant"]
        u = [_digit_norm(m.get("content")) for m in msgs if m.get("role") == "user"]
        a_all.extend(a)
        u_all.extend(u)
        if len(a) != len(set(a)):
            internal_dup += 1
    if not a_all or not u_all:
        raise ValueError("样本为空")
    return {
        "n_dialogues": len(dialogues),
        "assistant_unique_rate": len(set(a_all)) / len(a_all),
        "user_unique_rate": len(set(u_all)) / len(u_all),
        "internal_dup_rate": internal_dup / len(dialogues),
        "message_min": min(len(d.get("messages") or []) for d in dialogues),
        "message_max": max(len(d.get("messages") or []) for d in dialogues),
    }


def _digit_norm(text) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", re.sub(r"\s+", "", str(text or "")))
