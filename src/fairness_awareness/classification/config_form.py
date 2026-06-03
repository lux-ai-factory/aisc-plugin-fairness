from typing import Any

from pydantic import BaseModel, Field, model_validator

from ..utils import Feature, FeatureType


FORM_UI_SCHEMA: dict[str, dict[str, Any]] = {
    "features": {
        "ui:options": {
            "orderable": False,
            "addable": False,
        },
        "items": {
            "ui:field": "LayoutGridField",
            "ui:layoutGrid": {
                "ui:row": {
                    "className": "row",
                    "children": [
                        {"ui:col": {"className": "col-4", "children": ["name"]}},
                        {"ui:col": {"className": "col-3", "children": ["min"]}},
                        {"ui:col": {"className": "col-3", "children": ["max"]}},
                        {
                            "ui:col": {
                                "className": "col-2",
                                "children": ["type"],
                            }
                        },
                    ],
                }
            },
        },
    },
}


class ConfigForm(BaseModel):
    target_feature: str | None = Field(default=None, title="Target Feature")
    date_feature: str | None = Field(default=None, title="Date Feature")
    interest_feature_1: str | None = Field(default=None, title="Feature of Interest")
    interest_feature_2: str | None = Field(default=None, title="Feature of Interest")
    interest_feature_3: str | None = Field(default=None, title="Feature of Interest")

    features: list[Feature] = Field(
        default_factory=list, description="List of features to use for prediction"
    )

    @model_validator(mode="after")
    def validate_special_features(self) -> "ConfigForm":
        self.target_feature = self.target_feature or None
        self.date_feature = self.date_feature or None
        self.interest_feature_1 = self.interest_feature_1 or None
        self.interest_feature_2 = self.interest_feature_2 or None
        self.interest_feature_3 = self.interest_feature_3 or None

        # Target feature validation
        if self.target_feature is None:
            # raise ValueError("Target feature must be specified.")
            return self

        target_feature = next(
            (f for f in self.features if f.name == self.target_feature), None
        )
        if target_feature is None:
            raise ValueError("Target feature must be one of the configured features.")

        # Date feature validation
        if self.date_feature is None:
            return self

        date_feature = next(
            (f for f in self.features if f.name == self.date_feature), None
        )
        if date_feature is None:
            raise ValueError("Date feature must be one of the configured features.")

        if date_feature.type != FeatureType.DATE:
            raise ValueError("Date feature must be of type Date.")

        if target_feature == date_feature:
            raise ValueError("Target feature must be different from date feature.")

        return self
