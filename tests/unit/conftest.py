import pytest


@pytest.fixture
def y_true_v1():
    return ["A"] * 10 + ["B"] * 20


@pytest.fixture
def y_pred_v1():
    return [v for _ in range(15) for v in ["A", "B"]]


@pytest.fixture
def labels_v1():
    return ["A", "B"]


@pytest.fixture
def expected_output_compute_metrics_v1():
    return {
        "accuracy": 0.5,
        "f1_macro": 0.48571428571428577,
        "precision_macro": 0.5,
        "recall_macro": 0.5,
    }


@pytest.fixture
def expected_output_compute_metrics_by_class_v1():
    return {
        "f1_by_class": {"A": 0.4, "B": 0.5714285714285715},
        "precision_by_class": {"A": 0.3333333333333333, "B": 0.6666666666666666},
        "recall_by_class": {"A": 0.5, "B": 0.5},
    }
