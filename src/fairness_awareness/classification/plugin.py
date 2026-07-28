from aisc_plugin_interface import Measure, MetricVisualization, ChartType

from .base_plugin import BaseClassificationFairnessPlugin
from ..model_input_provider import OnnxModelSession
from ..utils import (
    check_features,
    parse_label_mappings,
    FeatureType,
)


class ClassificationFairnessPlugin(BaseClassificationFairnessPlugin):
    """Classification Fairness plugin."""

    plugin_name = "Classification Fairness"
    ui_icon = "balance"

    # ========================== Evaluation ==========================

    METRIC_NAMES = ["accuracy", "f1", "precision", "recall", "mcc"]

    def evaluate(self, config_data: dict) -> dict[str, list[dict]]:
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

        config = self.validate_config_form_data(config_data)
        target_feature_name = config.target_feature
        column_feature_names, interest_feature_names, failure = check_features(config, self.logger)

        if failure:
            return {name: [] for name in self.METRIC_NAMES}

        parsed_mappings = parse_label_mappings(config.label_mappings)
        for feature in config.features:
            if feature.name in parsed_mappings:
                feature.label_mapping = parsed_mappings[feature.name]

        self.logger.debug(
            "Prepared %d features (%d used as features of interest)",
            len(column_feature_names),
            len(interest_feature_names),
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
        baseline_precision = precision_score(y_true, y_pred, zero_division=0, average="weighted")
        baseline_recall = recall_score(y_true, y_pred, zero_division=0, average="weighted")
        baseline_mcc = (matthews_corrcoef(y_true, y_pred) + 1) / 2

        output = {name: [] for name in self.METRIC_NAMES}

        for feature in config.features:
            if feature is None or feature.name not in interest_feature_names:
                continue

            feature_name = feature.name

            # Baseline scores
            baseline_scores = {
                "accuracy": baseline_accuracy,
                "f1": baseline_f1,
                "precision": baseline_precision,
                "recall": baseline_recall,
                "mcc": baseline_mcc,
            }
            for metric_name in self.METRIC_NAMES:
                output[metric_name].append({
                    "score": baseline_scores[metric_name],
                    "dimensions": {
                        "display_name": f"{feature_name}_{metric_name}",
                        "group": "baseline",
                        "feature": feature_name,
                    },
                })

            for group in df_test[feature.name].unique():
                self.logger.debug(f"Calculating metrics for group '{group}' in feature '{feature.name}'")

                mask = df_test[feature.name] == group
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                if feature.label_mapping:
                    group_name = feature.label_mapping.get(str(group), str(group))
                elif feature.type == FeatureType.INTEGER:
                    group_name = f"group_{group}"
                else:
                    group_name = str(group)

                group_scores = {
                    "accuracy": accuracy_score(y_true_group, y_pred_group),
                    "f1": f1_score(y_true_group, y_pred_group, zero_division=0, average="weighted"),
                    "precision": precision_score(y_true_group, y_pred_group, zero_division=0, average="weighted"),
                    "recall": recall_score(y_true_group, y_pred_group, zero_division=0, average="weighted"),
                    "mcc": (matthews_corrcoef(y_true_group, y_pred_group) + 1) / 2,
                }

                for metric_name in self.METRIC_NAMES:
                    output[metric_name].append({
                        "score": group_scores[metric_name],
                        "dimensions": {
                            "display_name": f"{feature_name}_{metric_name}",
                            "group": group_name,
                            "feature": feature_name,
                        },
                    })

        self.logger.info("Classification fairness evaluation completed")

        return output

    # ========================== Metrics ==========================

    def get_metrics(self) -> list[str]:
        return list(self.METRIC_NAMES)

    def export_metrics(self, evaluation_output: dict) -> list[Measure]:
        results: list[Measure] = []
        for metric_name, entries in evaluation_output.items():
            for entry in entries:
                results.append(
                    Measure(
                        name=metric_name,
                        score=entry["score"],
                        dimensions=entry["dimensions"],
                    )
                )
        return results

    # ========================== Visualization ==========================

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.config_type.model_validate(config_data)
        _, interest_feature_names, failure = check_features(config, self.logger)
        if failure:
            return []
        return [
            MetricVisualization(
                chart_type=ChartType.RADAR,
                metrics=list(self.METRIC_NAMES),
                title=f"Fairness by {feature_name}",
                description=f"Per-group metrics for feature '{feature_name}'",
                filter_dimensions={"feature": feature_name},
            )
            for feature_name in interest_feature_names
        ]
