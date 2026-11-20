import math

from lidds_gen.evaluation import compute_metrics, metrics_by_length, select_threshold


def test_f1_threshold_separates_perfectly():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.8, 0.9]
    tau = select_threshold(y, p, "f1")
    m = compute_metrics(y, p, tau)
    assert m["f1"] == 1.0 and m["fpr"] == 0.0


def test_single_class_metrics_are_nan():
    m = compute_metrics([0, 0], [0.1, 0.7], 0.5)
    assert math.isnan(m["auroc"]) and m["fp"] == 1


def test_by_length_keys():
    out = metrics_by_length([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [50, 50, 200, 200], 0.5)
    assert set(out) == {50, 200}
