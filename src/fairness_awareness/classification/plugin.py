from typing import Any

from aisc_plugin_interface import metric, Measure, MetricVisualization, ChartType

from .base_plugin import BaseClassificationFairnessPlugin
from ..model_input_provider import OnnxModelSession
from ..utils import (
    export_metric,
    check_features,
    FeatureType,
)


class ClassificationFairnessPlugin(BaseClassificationFairnessPlugin):
    """Classification Fairness plugin."""

    plugin_name = "Classification Fairness"
    ui_icon = "balance"

    # ========================== Evaluation ==========================

    def evaluate(self, config_data: dict) -> dict[str, dict[str, Any]]:
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
        column_feature_names, interest_feature_names, failure = check_features(config, self.logger)
 
        if failure:
            return {}

        self.logger.debug(
            "Prepared %d features (%d used as features of interest)",
            len(column_feature_names),
            len(interest_feature_names.values()),
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


        # metric name holding dicts with score(s) and description(s)
        metrics: dict[str, dict[str, Any]] = {}

        baseline_accuracy = accuracy_score(y_true, y_pred)
        baseline_f1 = f1_score(y_true, y_pred, zero_division=0, average="weighted")
        baseline_precision = precision_score(y_true, y_pred, zero_division=0, average="weighted")
        baseline_recall = recall_score(y_true, y_pred, zero_division=0, average="weighted")
        baseline_mcc = (matthews_corrcoef(y_true, y_pred) + 1) / 2
    
        # For each selected feature
        for feature in config.features:
            if feature is None or feature.name not in interest_feature_names.keys():
                continue

            metric_name_prefix = interest_feature_names[feature.name]

            metrics[f"{metric_name_prefix}_accuracy"] = {"score": [baseline_accuracy], "description": ["baseline"]}
            metrics[f"{metric_name_prefix}_f1"] = {"score": [baseline_f1], "description": ["baseline"]}
            metrics[f"{metric_name_prefix}_precision"] = {"score": [baseline_precision], "description": ["baseline"]}
            metrics[f"{metric_name_prefix}_recall"] = {"score": [baseline_recall], "description": ["baseline"]}
            metrics[f"{metric_name_prefix}_mcc"] = {"score": [baseline_mcc], "description": ["baseline"]}

            # For each group in feature
            for group in df_test[feature.name].unique():
                self.logger.debug(f"Calculating metrics for group '{group}' in feature '{feature.name}'")

                mask = df_test[feature.name] == group
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                group_name = f"group_{group}" if feature.type == FeatureType.INTEGER else str(group)
                
                # Store metrics with descriptions
                group_accuracy = accuracy_score(y_true_group, y_pred_group)
                metrics[f"{metric_name_prefix}_accuracy"]["score"].append(group_accuracy)
                metrics[f"{metric_name_prefix}_accuracy"]["description"].append(group_name)

                group_f1 = f1_score(y_true_group, y_pred_group, zero_division=0, average="weighted")
                metrics[f"{metric_name_prefix}_f1"]["score"].append(group_f1)
                metrics[f"{metric_name_prefix}_f1"]["description"].append(group_name)

                group_precision = precision_score(y_true_group, y_pred_group, zero_division=0, average="weighted")
                metrics[f"{metric_name_prefix}_precision"]["score"].append(group_precision)
                metrics[f"{metric_name_prefix}_precision"]["description"].append(group_name)

                group_recall = recall_score(y_true_group, y_pred_group, zero_division=0, average="weighted")
                metrics[f"{metric_name_prefix}_recall"]["score"].append(group_recall)
                metrics[f"{metric_name_prefix}_recall"]["description"].append(group_name)

                group_mcc = (matthews_corrcoef(y_true_group, y_pred_group) + 1) / 2
                metrics[f"{metric_name_prefix}_mcc"]["score"].append(group_mcc)
                metrics[f"{metric_name_prefix}_mcc"]["description"].append(group_name)

        self.logger.info("Classification fairness evaluation completed")

        return metrics

    # ========================== Metrics ==========================

    # Interest feature 1
    @metric("interest_feature_1_accuracy")
    def interest_feature_1_accuracy_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_1_accuracy", evaluation_output)
    
    @metric("interest_feature_1_f1")
    def interest_feature_1_f1_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_1_f1", evaluation_output)
    
    @metric("interest_feature_1_precision")
    def interest_feature_1_precision_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_1_precision", evaluation_output)

    @metric("interest_feature_1_recall")
    def interest_feature_1_recall_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_1_recall", evaluation_output)

    @metric("interest_feature_1_mcc")
    def interest_feature_1_mcc_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_1_mcc", evaluation_output)

    # Interest feature 2
    @metric("interest_feature_2_accuracy")
    def interest_feature_2_accuracy_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_2_accuracy", evaluation_output)

    @metric("interest_feature_2_f1")
    def interest_feature_2_f1_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_2_f1", evaluation_output)
    
    @metric("interest_feature_2_precision")
    def interest_feature_2_precision_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_2_precision", evaluation_output)

    @metric("interest_feature_2_recall")
    def interest_feature_2_recall_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_2_recall", evaluation_output)

    @metric("interest_feature_2_mcc")
    def interest_feature_2_mcc_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_2_mcc", evaluation_output)

    # Interest feature 3
    @metric("interest_feature_3_accuracy")
    def interest_feature_3_accuracy_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_3_accuracy", evaluation_output)

    @metric("interest_feature_3_f1")
    def interest_feature_3_f1_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_3_f1", evaluation_output)
    
    @metric("interest_feature_3_precision")
    def interest_feature_3_precision_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_3_precision", evaluation_output)

    @metric("interest_feature_3_recall")
    def interest_feature_3_recall_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_3_recall", evaluation_output)

    @metric("interest_feature_3_mcc")
    def interest_feature_3_mcc_metric(self, evaluation_output: dict) -> list[Measure]:
        return export_metric("interest_feature_3_mcc", evaluation_output)

    # ========================== Visualization ==========================

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        return [
            MetricVisualization(
                chart_type=ChartType.RADAR,
                metrics=[
                    "interest_feature_1_accuracy",
                    "interest_feature_1_f1",
                    "interest_feature_1_precision",
                    "interest_feature_1_recall",
                    "interest_feature_1_mcc",
                ],
            ),
            MetricVisualization(
                chart_type=ChartType.RADAR,
                metrics=[
                    "interest_feature_2_accuracy",
                    "interest_feature_2_f1",
                    "interest_feature_2_precision",
                    "interest_feature_2_recall",
                    "interest_feature_2_mcc",
                ],
            ),
            MetricVisualization(
                chart_type=ChartType.RADAR,
                metrics=[
                    "interest_feature_3_accuracy",
                    "interest_feature_3_f1",
                    "interest_feature_3_precision",
                    "interest_feature_3_recall",
                    "interest_feature_3_mcc",
                ],
            ),
        ]
