from aisc_plugin_interface import Measure, MetricVisualization, ChartType

from .base_plugin import BaseClassificationFairnessPlugin
from ..model_input_provider import OnnxModelSession
from ..utils import (
    check_features,
    parse_label_mappings,
    FeatureType,
)


@staticmethod
def _group_name(feature, group_val):
    if feature.label_mapping:
        return feature.label_mapping.get(str(group_val), str(group_val))
    if feature.type == FeatureType.INTEGER:
        return f"group_{group_val}"
    return str(group_val)

@staticmethod
def _per_class_rates(cm):
    n = cm.shape[0]
    tpr, fpr, fnr, tnr = [], [], [], []
    for c in range(n):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnr.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
        tnr.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return tpr, fpr, fnr, tnr

@staticmethod
def _macro_avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

@staticmethod
def _disparity(group_metrics, metric_key, gnames):
    vals = [group_metrics[g][metric_key] for g in gnames]
    return max(vals) - min(vals)


class ClassificationFairnessPlugin(BaseClassificationFairnessPlugin):
    """Classification Fairness plugin."""

    plugin_name = "Classification Fairness"
    ui_icon = "balance"

    # Per-group metrics (one entry per group + baseline)
    METRIC_NAMES = [
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "precision",
        "f1",
        "recall",
        "specificity",
        "fpr",
        "fnr", 
    ]

    # Disparity metrics (single value per feature, comparing across groups)
    DISPARITY_METRIC_NAMES = [
        "disparity_demographic_parity",
        "disparity_disparate_impact",
        "disparity_accuracy",
        "disparity_balanced_accuracy",
        "disparity_mcc",
        "disparity_precision",
        "disparity_f1",
        "disparity_recall",
        "disparity_specificity",
        "disparity_fpr",
        "disparity_fnr",
    ]

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
            balanced_accuracy_score,
            confusion_matrix,
        )

        self.logger.info("Starting classification fairness evaluation")

        config = self.validate_config_form_data(config_data)
        target_feature_name = config.target_feature
        column_feature_names, interest_feature_names, failure = check_features(config, self.logger)

        if failure:
            all_names = self.METRIC_NAMES + self.DISPARITY_METRIC_NAMES
            return {name: [] for name in all_names}

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

        all_metric_names = self.METRIC_NAMES + self.DISPARITY_METRIC_NAMES
        output = {name: [] for name in all_metric_names}

        # Compute metrics for each feature of interest
        for feature in config.features:
            if feature is None or feature.name not in interest_feature_names:
                continue

            feature_name = feature.name
            groups = df_test[feature.name].unique()
            group_metrics = {}
            group_pred_dist = {}

            # Baseline (full dataset)
            cm_full = confusion_matrix(y_true, y_pred)
            n_classes_full = cm_full.shape[0]
            _, fpr_full, fnr_full, tnr_full = _per_class_rates(cm_full)

            baseline_scores = {
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "mcc": (matthews_corrcoef(y_true, y_pred) + 1) / 2,
                "precision": precision_score(y_true, y_pred, zero_division=0, average="weighted"),
                "f1": f1_score(y_true, y_pred, zero_division=0, average="weighted"),
                "recall": recall_score(y_true, y_pred, zero_division=0, average="weighted"),
                "specificity": _macro_avg(tnr_full),
                "fpr": _macro_avg(fpr_full),
                "fnr": _macro_avg(fnr_full),
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

            # Per-group metrics
            for group_val in groups:
                mask = df_test[feature.name] == group_val
                y_true_g = y_true[mask]
                y_pred_g = y_pred[mask]

                gname = _group_name(feature, group_val)
                cm_g = confusion_matrix(y_true_g, y_pred_g)
                _, fpr_list, fnr_list, tnr_list = _per_class_rates(cm_g)

                # Full predicted-class distribution for this group (used for demographic parity)
                group_pred_dist[gname] = np.bincount(y_pred_g, minlength=n_classes_full).astype(float) / len(y_pred_g)
                group_metrics[gname] = {
                    "accuracy": accuracy_score(y_true_g, y_pred_g),
                    "balanced_accuracy": balanced_accuracy_score(y_true_g, y_pred_g),
                    "mcc": (matthews_corrcoef(y_true_g, y_pred_g) + 1) / 2,
                    "precision": precision_score(y_true_g, y_pred_g, zero_division=0, average="weighted"),
                    "f1": f1_score(y_true_g, y_pred_g, zero_division=0, average="weighted"),
                    "recall": recall_score(y_true_g, y_pred_g, zero_division=0, average="weighted"),
                    "specificity": _macro_avg(tnr_list),
                    "fpr": _macro_avg(fpr_list),
                    "fnr": _macro_avg(fnr_list),
                }

                for metric_name in self.METRIC_NAMES:
                    output[metric_name].append({
                        "score": group_metrics[gname][metric_name],
                        "dimensions": {
                            "display_name": f"{feature_name}_{metric_name}",
                            "group": gname,
                            "feature": feature_name,
                        },
                    })

            # Disparity metrics require at least 2 groups to compare
            if len(group_metrics) < 2:
                for disp_name in self.DISPARITY_METRIC_NAMES:
                    output[disp_name].append({
                        "score": 0.0,
                        "dimensions": {
                            "display_name": f"{feature_name}_{disp_name}",
                            "feature": feature_name,
                        },
                    })
                continue

            gnames = list(group_metrics.keys())
            dists = np.array([group_pred_dist[g] for g in gnames])

            if n_classes_full == 2:
                # Binary: standard demographic parity uses the positive class (class 1)
                pos_rates = dists[:, 1]
                demo_parity = float(pos_rates.max() - pos_rates.min())
                impact_ratio = float(pos_rates.min() / pos_rates.max()) if pos_rates.max() > 0 else 0.0
            else:
                # Multi-class: worst-case class disparity across all classes
                class_disparities = dists.max(axis=0) - dists.min(axis=0)
                demo_parity = float(class_disparities.max())
                class_ratios = np.where(
                    dists.max(axis=0) > 0,
                    dists.min(axis=0) / dists.max(axis=0),
                    1.0,
                )
                impact_ratio = float(class_ratios.min())

            disparity_scores = {
                "disparity_demographic_parity": demo_parity,
                "disparity_disparate_impact": impact_ratio,
                "disparity_accuracy": _disparity(group_metrics, "accuracy", gnames),
                "disparity_balanced_accuracy": _disparity(group_metrics, "balanced_accuracy", gnames),
                "disparity_mcc": _disparity(group_metrics, "mcc", gnames),
                "disparity_precision": _disparity(group_metrics, "precision", gnames),
                "disparity_f1": _disparity(group_metrics, "f1", gnames),
                "disparity_recall": _disparity(group_metrics, "recall", gnames),
                "disparity_specificity": _disparity(group_metrics, "specificity", gnames),
                "disparity_fpr": _disparity(group_metrics, "fpr", gnames),
                "disparity_fnr": _disparity(group_metrics, "fnr", gnames),
            }

            for disp_name, score in disparity_scores.items():
                output[disp_name].append({
                    "score": score,
                    "dimensions": {
                        "display_name": f"{feature_name}_{disp_name}",
                        "feature": feature_name,
                    },
                })

        self.logger.info("Classification fairness evaluation completed")
        return output

    # ========================== Metrics ==========================

    def get_metrics(self) -> list[str]:
        return self.METRIC_NAMES + self.DISPARITY_METRIC_NAMES

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

        visualizations = []

        for feature_name in interest_feature_names:
            # Radar: original per-group classification metrics
            visualizations.append(
                MetricVisualization(
                    chart_type=ChartType.RADAR,
                    metrics=self.METRIC_NAMES,
                    title=f"Classification Metrics by {feature_name}",
                    description=f"Per-group classification quality metrics for feature '{feature_name}'",
                    filter_dimensions={"feature": [feature_name]},
                    group_by_dimensions=["group"],
                )
            )

            # Bar: disparity metrics
            visualizations.append(
                MetricVisualization(
                    chart_type=ChartType.BARS,
                    metrics=self.DISPARITY_METRIC_NAMES,
                    title=f"Fairness Disparities by {feature_name}",
                    description=f"Fairness disparity metrics for feature '{feature_name}' (lower is fairer)",
                    filter_dimensions={"feature": [feature_name]},
                )
            )

        return visualizations
