"""首条 user 屏蔽（业务随机开场句）的单元测试。"""

from agent.eval.preprocess import OPENING_PLACEHOLDER, mask_dialogues, mask_opening_user


def _dlg():
    return {"messages": [
        {"role": "system", "content": "<姓名>张三"},
        {"role": "user", "content": "你们要是给我三方打电话我就投诉你们。"},
        {"role": "assistant", "content": "下午好，请问是张三先生吗？", "loss": "True"},
        {"role": "user", "content": "是的。"},
        {"role": "assistant", "content": "关于还款……", "loss": "True"},
    ]}


def test_masks_first_user_only():
    out = mask_opening_user(_dlg())
    msgs = out["messages"]
    assert msgs[1]["content"] == OPENING_PLACEHOLDER
    assert msgs[3]["content"] == "是的。"          # 后续 user 不动
    assert len(msgs) == 5                          # 条数不变
    assert msgs[0]["content"] == "<姓名>张三"       # system 不动


def test_does_not_mutate_input():
    src = _dlg()
    mask_opening_user(src)
    assert src["messages"][1]["content"].startswith("你们要是")


def test_idempotent():
    once = mask_opening_user(_dlg())
    twice = mask_opening_user(once)
    assert once == twice


def test_no_user_returns_copy():
    d = {"messages": [{"role": "system", "content": "x"},
                      {"role": "assistant", "content": "y"}]}
    out = mask_opening_user(d)
    assert out == d and out is not d


def test_mask_dialogues_batch():
    out = mask_dialogues([_dlg(), _dlg()])
    assert len(out) == 2
    assert all(o["messages"][1]["content"] == OPENING_PLACEHOLDER for o in out)
