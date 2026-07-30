from aisc_plugin_interface import metric, Measure, MetricVisualization, ChartType

from .base_plugin import BaseClassificationFairnessPlugin
from ..model_input_provider import OnnxModelSession
from ..utils import check_features, FeatureType


class ClassificationFairnessPlugin(BaseClassificationFairnessPlugin):
    """Classification Fairness plugin."""

    plugin_name = "Classification Fairness"
    ui_icon = "balance"

    METRIC_NAMES = ["accuracy", "f1", "precision", "recall", "mcc"]

    # ========================== Evaluation ==========================

    def evaluate(self, config_data: dict) -> dict[str, list[Measure]]:
        measures: dict[str, list[Measure]] = dict.fromkeys(self.METRIC_NAMES, [])

        from onnxruntime import InferenceSession
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            matthews_corrcoef,
        )

        self.logger.info("Starting classification fairness evaluation")

        # Config setup
        config = self.validate_config_form_data(config_data)
        target_feature_name = config.target_feature
        column_feature_names, failure = check_features(config, self.logger)

        if failure:
            return {name: [] for name in self.METRIC_NAMES}

        self.logger.debug(
            "Prepared %d features",
            len(column_feature_names),
        )

        # Dataset
        try:
            df_test = self.get_input_data("test-dataset")
        except Exception:
            self.logger.exception("Failed to load test dataset")
            raise
        assert isinstance(df_test, pd.DataFrame)
        x_test_np = df_test[column_feature_names].to_numpy().astype(np.float32)
        y_true = df_test[target_feature_name].to_numpy().astype(np.int64)

        # ONNX runtime
        try:
            session = self.get_input_data("model")
        except Exception:
            self.logger.exception("Failed to load ONNX model")
            raise
        assert isinstance(session, InferenceSession)
        model_session = OnnxModelSession(session)

        y_pred_proba = model_session.predict(x_test_np, probabilities=True)
        y_pred = np.argmax(y_pred_proba, axis=1)

        baseline_accuracy = accuracy_score(y_true, y_pred)
        baseline_f1 = f1_score(y_true, y_pred, zero_division=0, average="weighted")
        baseline_precision = precision_score(
            y_true, y_pred, zero_division=0, average="weighted"
        )
        baseline_recall = recall_score(
            y_true, y_pred, zero_division=0, average="weighted"
        )
        baseline_mcc = (matthews_corrcoef(y_true, y_pred) + 1) / 2

        baseline_scores = {
            "accuracy": baseline_accuracy,
            "f1": baseline_f1,
            "precision": baseline_precision,
            "recall": baseline_recall,
            "mcc": baseline_mcc,
        }

        # ADDED FIX: Use metric_name instead of hardcoding "accuracy"
        for metric_name in self.METRIC_NAMES:
            measures[metric_name].append(
                Measure(
                    name=metric_name,
                    score=baseline_scores[metric_name],
                    dimensions={"feature_group": "baseline", "feature": "N/A"},
                    unit="percentage",
                )
            )

        # For each selected feature
        for column_feature_name in column_feature_names:
            feature = next(
                (f for f in config.features if f.name == column_feature_name), None
            )
            if feature is None:
                continue

            unique_val_count = df_test[feature.name].nunique()
            if feature.type == FeatureType.CATEGORICAL or unique_val_count < 4:
                groups = df_test[feature.name].astype(str)
            else:
                try:
                    binned = pd.qcut(df_test[feature.name], q=4, duplicates="drop")

                    unique_bins = sorted(binned.dropna().unique())
                    num_bins = len(unique_bins)

                    if num_bins == 4:
                        prefix = "Q"
                    elif num_bins == 3:
                        prefix = "Tertile "
                    elif num_bins == 2:
                        prefix = "Half "
                    else:
                        prefix = "Bin "

                    interval_mapping = {}
                    for i, interval in enumerate(unique_bins):
                        left = round(interval.left, 2)
                        right = round(interval.right, 2)
                        interval_mapping[interval] = (
                            f"{prefix}{i + 1} ({left} - {right})"
                        )

                    groups = binned.map(interval_mapping).astype(str)
                except ValueError:
                    groups = df_test[feature.name].astype(str)

            for group in sorted(groups.unique()):
                self.logger.debug(
                    f"Calculating metrics for group '{group}' in feature '{feature.name}'"
                )

                mask = groups == group
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                if len(y_true_group) == 0:
                    continue

                group_scores = {
                    "accuracy": accuracy_score(y_true_group, y_pred_group),
                    "f1": f1_score(
                        y_true_group,
                        y_pred_group,
                        zero_division=0,
                        average="weighted",
                    ),
                    "precision": precision_score(
                        y_true_group,
                        y_pred_group,
                        zero_division=0,
                        average="weighted",
                    ),
                    "recall": recall_score(
                        y_true_group,
                        y_pred_group,
                        zero_division=0,
                        average="weighted",
                    ),
                    "mcc": (matthews_corrcoef(y_true_group, y_pred_group) + 1) / 2,
                }

                for metric_name in self.METRIC_NAMES:
                    measures[metric_name].append(
                        Measure(
                            name=metric_name,
                            score=group_scores[metric_name],
                            dimensions={
                                "feature_group": str(group),
                                "feature": feature.name,
                            },
                            unit="percentage",
                        )
                    )

        self.logger.info("Classification fairness evaluation completed")

        return measures

    # ========================== Metrics ==========================

    @metric("accuracy")
    def accuracy_metric(
        self, evaluation_output: dict[str, list[Measure]]
    ) -> list[Measure]:
        return evaluation_output["accuracy"]

    @metric("f1")
    def f1_metric(self, evaluation_output: dict[str, list[Measure]]) -> list[Measure]:
        return evaluation_output["f1"]

    @metric("precision")
    def recision_metric(
        self, evaluation_output: dict[str, list[Measure]]
    ) -> list[Measure]:
        return evaluation_output["precision"]

    @metric("recall")
    def recall_metric(
        self, evaluation_output: dict[str, list[Measure]]
    ) -> list[Measure]:
        return evaluation_output["recall"]

    @metric("mcc")
    def mcc_metric(self, evaluation_output: dict[str, list[Measure]]) -> list[Measure]:
        return evaluation_output["mcc"]

    # ========================== Visualization ==========================

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        metric_visualizations: list[MetricVisualization] = []

        metric_visualization = MetricVisualization(
            title=f"Baseline",
            chart_type=ChartType.RADAR,
            metrics=[
                "accuracy",
                "f1",
                "precision",
                "recall",
                "mcc",
            ],
            filter_dimensions={"feature": ["N/A"]},
        )
        metric_visualizations.append(metric_visualization)

        for feature in config.features:
            if feature.name in [config.target_feature, config.date_feature]:
                continue

            metric_visualization = MetricVisualization(
                title=f"Feature ({feature.name})",
                chart_type=ChartType.RADAR,
                metrics=[
                    "accuracy",
                    "f1",
                    "precision",
                    "recall",
                    "mcc",
                ],
                filter_dimensions={"feature": [feature.name, "N/A"]},
                group_by_dimension_keys=["feature_group"],
            )
            metric_visualizations.append(metric_visualization)

        return metric_visualizations
