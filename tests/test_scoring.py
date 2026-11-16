import numpy as np
import pytest

from ft.stats import bootstrap_ci, paired_diff_ci, verdict_phrase, verdict_word
from ft.tasks.base import ScoreRow
from ft.tasks.classify import ClassifyTask, normalize_label

LABELS = ["card_arrival", "lost_card", "Top Up"]


@pytest.fixture
def task(make_cfg):
    return ClassifyTask(make_cfg(), LABELS)


def test_normalize():
    assert normalize_label("  Lost_Card \n") == "lost_card"
    assert normalize_label("top   up") == "top up"


@pytest.mark.parametrize(
    "raw,structured,expected,valid",
    [
        ('{"label": "lost_card"}', True, "lost_card", True),
        ('{"label": "LOST_CARD"}', True, "lost_card", True),  # enum case may change
        ("top  UP", False, "Top Up", True),
        (" card_arrival\n", False, "card_arrival", True),
        ("card arrival", False, None, False),
        ('{"label": "nope"}', True, None, False),
        ("", False, None, False),
        (None, False, None, False),
    ],
)
def test_parse(task, raw, structured, expected, valid):
    p = task.parse(raw, structured=structured)
    assert (p.value, p.valid) == (expected, valid)


def test_baseline_prompt_lists_labels_and_schema(task):
    p = task.baseline_prompt(context="I run a small bakery.")
    assert "lost_card" in p.system and "bakery" in p.system
    assert p.json_schema["properties"]["label"]["enum"] == LABELS
    assert p.json_schema["additionalProperties"] is False


def test_tuned_prompt_is_instruction_only(task):
    p = task.tuned_prompt()
    assert p.system == task.instruction and "lost_card" not in p.system and p.json_schema is None


def _rows(pairs):
    return [
        ScoreRow(id=str(i), input=f"in {i}", target=t, parsed=p, valid=p is not None) for i, (t, p) in enumerate(pairs)
    ]


def test_accuracy_and_macro_f1_by_hand(task):
    # a=card_arrival, b=lost_card
    rows = _rows(
        [
            ("card_arrival", "card_arrival"),
            ("card_arrival", "lost_card"),
            ("lost_card", "lost_card"),
            ("lost_card", None),
        ]
    )
    s = task.scorer(rows)
    idx = np.arange(4)
    assert s.metric_fn("accuracy")(idx) == pytest.approx(50.0)
    # card_arrival: tp1 fp0 fn1 → f1 = 2/3 ; lost_card: tp1 fp1 fn1 → f1 = 1/2
    assert s.metric_fn("macro_f1")(idx) == pytest.approx((2 / 3 + 1 / 2) / 2 * 100)
    d = s.details()
    assert d["invalid_rate"] == 25.0
    pairs = {(c["true"], c["pred"]) for c in d["confusions"]}
    assert pairs == {("card_arrival", "lost_card"), ("lost_card", "(invalid)")}


def test_target_unseen_in_train_counts_as_wrong(task):
    s = task.scorer(_rows([("brand_new_label", "lost_card"), ("lost_card", "lost_card")]))
    assert s.metric_fn("accuracy")(np.arange(2)) == 50.0


def test_bootstrap_ci_brackets_point(task):
    rng = np.random.default_rng(0)
    pairs = [("lost_card", "lost_card" if rng.random() < 0.8 else "card_arrival") for _ in range(500)]
    s = task.scorer(_rows(pairs))
    fn = s.metric_fn("accuracy")
    lo, hi = bootstrap_ci(fn, 500, seed=1)
    assert lo < fn(np.arange(500)) < hi and hi - lo < 10


def test_paired_verdict_words(task):
    truth = ["lost_card"] * 400
    good = _rows([(t, t) for t in truth])
    worse = _rows([(t, t if i % 5 else "card_arrival") for i, t in enumerate(truth)])  # 80%
    same = _rows([(t, t if i % 5 else "card_arrival") for i, t in enumerate(truth)])
    fa = task.scorer(good).metric_fn("accuracy")
    fb = task.scorer(worse).metric_fn("accuracy")
    fc = task.scorer(same).metric_fn("accuracy")
    assert verdict_word(*paired_diff_ci(fa, fb, 400)) == "beat"
    assert verdict_word(*paired_diff_ci(fb, fa, 400)) == "fell short"
    assert verdict_word(*paired_diff_ci(fb, fc, 400)) == "tied"


def test_inconclusive_differences_say_ahead_or_behind_never_beat():
    # The Enron case: 5.3 points behind with a CI that just reaches 0.
    assert verdict_word(-5.31, -9.95, 0.48) == "behind"
    assert verdict_word(1.2, -0.8, 3.1) == "ahead"
    assert verdict_phrase("behind", -5.31, "gpt-6-luna") == (
        "behind gpt-6-luna by 5.3 points, not statistically significant"
    )
    assert verdict_phrase("beat", 4.2) == "beat by 4.2 points"


def test_collapse_check_flags_constant_output_but_not_a_skewed_classifier():
    from ft.stages.evaluate import collapse_check

    blank = '{"total": ""}'
    c = collapse_check([blank] * 20, [{"total": str(i)} for i in range(20)])
    assert c["collapsed"] and c["distinct_outputs"] == 1 and "ignoring its input" in c["message"]
    # 70% of targets are "billing"; predicting it 80% of the time is skewed, not collapsed
    targets = ["billing"] * 14 + ["refund"] * 6
    outputs = ["billing"] * 16 + ["refund"] * 4
    assert not collapse_check(outputs, targets)["collapsed"]
    assert not collapse_check(["x"] * 5, ["a", "b", "c", "d", "e"])["collapsed"]  # too few to judge
