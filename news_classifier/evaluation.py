import evaluate
import pandas as pd
import numpy as np
from sklearn import metrics
import plotly.figure_factory as ff


def compute_metrics_huggingface(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": {"name": "accuracy", "args": {}},
        "f1_macro": {"name": "f1", "args": {"average": "macro"}},
        "precision_macro": {"name": "precision", "args": {"average": "macro"}},
    }

    return {
        metric: evaluate.load(d["name"]).compute(
            predictions=predictions, references=labels, **d["args"]
        )[d["name"]]
        for metric, d in metrics.items()
    }


def compute_metrics(y_true, y_pred, labels):
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "precision_macro": metrics.precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": metrics.recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }


def compute_metrics_by_class(y_true, y_pred, labels):
    return {
        "f1_by_class": {
            c: v
            for c, v in zip(
                labels,
                metrics.f1_score(
                    y_true, y_pred, labels=labels, average=None, zero_division=0
                ),
            )
        },
        "precision_by_class": {
            c: v
            for c, v in zip(
                labels,
                metrics.precision_score(
                    y_true, y_pred, labels=labels, average=None, zero_division=0
                ),
            )
        },
        "recall_by_class": {
            c: v
            for c, v in zip(
                labels,
                metrics.recall_score(
                    y_true, y_pred, labels=labels, average=None, zero_division=0
                ),
            )
        },
    }


def build_confusion_matrix(y_true, y_pred, labels):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(confusion_matrix, index=labels, columns=labels)


def get_fig_heatmap(confusion_matrix):
    z = confusion_matrix.values

    x = list(confusion_matrix.columns)
    y = list(confusion_matrix.index)

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z_text, colorscale="Viridis"
    )

    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200), autosize=False)

    # add colorbar
    fig["data"][0]["showscale"] = True
    return fig
