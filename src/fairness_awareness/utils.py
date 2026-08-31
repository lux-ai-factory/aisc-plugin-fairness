import uuid
from enum import Enum
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field, field_serializer


class _HasMetricNames(Protocol):
    @classmethod
    def metric_names(cls) -> list[str]: ...


T = TypeVar("T", bound=_HasMetricNames)


class FeatureType(str, Enum):
    INTEGER = "Integer"
    FLOAT = "Float"
    CATEGORICAL = "Categorical"
    DATE = "Date"


class Feature(BaseModel):
    pid: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(...)
    min: float = Field(...)
    max: float = Field(...)
    type: FeatureType = Field(...)
    label_mapping: dict[str, str] | None = None

    @field_serializer("pid")
    def serialize_pid(self, pid: uuid.UUID | None) -> str | None:
        return str(pid) if pid is not None else None


# ==================== Data Utils ====================


def check_features(config, logger) -> tuple[list[str], list[str], bool]:
    """Validates features and returns (feature_names, interest_feature_names, failure_flag)."""
    
    # Target must be of type INTEGER or CATEGORICAL
    target_feature_name = config.target_feature
    target_feature = next((f for f in config.features if f.name == target_feature_name), None)
    if target_feature is None:
        logger.error("Target feature not found in features list.")
        return ([], [], True)
    if target_feature.type not in (FeatureType.INTEGER, FeatureType.CATEGORICAL):
        logger.error("Target feature '%s' must be of type INTEGER or CATEGORICAL.", target_feature_name)
        return ([], [], True)

    # Date feature must be of type DATE if specified
    date_feature_name = config.date_feature
    if date_feature_name is not None:
        date_feature = next((f for f in config.features if f.name == date_feature_name), None)
        if date_feature is None:
            logger.error("Date feature not found in features list.")
            return ([], [], True)
        if date_feature.type != FeatureType.DATE:
            logger.error("Date feature '%s' must be of type DATE.", date_feature_name)
            return ([], [], True)
    
    # Features of interest must be of type INTEGER or CATEGORICAL if specified
    interest_feature_names = config.interest_features
    if not interest_feature_names:
        logger.error("At least one interest feature must be specified.")
        return ([], [], True)
    for feature_name in interest_feature_names:
        feature = next((f for f in config.features if f.name == feature_name), None)
        if feature is None:
            logger.error("Interest feature '%s' not found in features list.", feature_name)
            return ([], [], True)
        if feature.type not in (FeatureType.INTEGER, FeatureType.CATEGORICAL):
            logger.error(
                "Interest feature '%s' must be of type INTEGER or CATEGORICAL.",
                feature_name,
            )
            return ([], [], True)

    # Prepare features
    column_feature_names = []
    failure = False

    # Verify feature types
    for feature in config.features:
        # Exclude target and date feature
        if feature.name in (target_feature_name, date_feature_name):
            continue

        # Check input feature types
        match feature.type:
            case FeatureType.DATE:
                logger.error(
                    "Feature '%s' is of type DATE but is not marked as the date feature.",
                    feature.name,
                )
                failure = True

        # Populate lists of features
        column_feature_names.append(feature.name)

    return (column_feature_names, interest_feature_names, failure)


def parse_label_mappings(mappings_str: str | None) -> dict[str, dict[str, str]]:
    """Parse a JSON string into {feature_name: {value_str: label_str}}."""
    if not mappings_str:
        return {}
    import json

    try:
        data = json.loads(mappings_str)
        if not isinstance(data, dict):
            return {}
        return {
            feat: {str(k): str(v) for k, v in mapping.items()}
            for feat, mapping in data.items()
            if isinstance(mapping, dict)
        }
    except (json.JSONDecodeError, ValueError):
        return {}
