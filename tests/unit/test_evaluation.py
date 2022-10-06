import pytest

from news_classifier.evaluation import compute_metrics, compute_metrics_by_class


@pytest.mark.parametrize(
    "y_true, y_pred, labels, expected_output",
    [
        (
            pytest.lazy_fixture("y_true_v1"),
            pytest.lazy_fixture("y_pred_v1"),
            pytest.lazy_fixture("labels_v1"),
            pytest.lazy_fixture("expected_output_compute_metrics_v1"),
        ),
    ],
)
def test_compute_metrics(y_true, y_pred, labels, expected_output):
    output = compute_metrics(y_true, y_pred, labels)
    assert output == expected_output


@pytest.mark.parametrize(
    "y_true, y_pred, labels, expected_output",
    [
        (
            pytest.lazy_fixture("y_true_v1"),
            pytest.lazy_fixture("y_pred_v1"),
            pytest.lazy_fixture("labels_v1"),
            pytest.lazy_fixture("expected_output_compute_metrics_by_class_v1"),
        ),
    ],
)
def test_compute_metrics_by_class(y_true, y_pred, labels, expected_output):
    output = compute_metrics_by_class(y_true, y_pred, labels)
    output == expected_output
