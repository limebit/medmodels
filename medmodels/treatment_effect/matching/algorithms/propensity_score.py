"""Propensity Score Matching.

This module provides functions for calculating propensity scores and executing
propensity score matching. The propensity score is the probability of being in the
treatment group given a set of covariates. Propensity score matching is a method for
matching treated and control units based on their propensity scores, ensuring that the
two groups are comparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, TypeAlias, Union

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from medmodels.treatment_effect.matching.algorithms.classic_distance_models import (
    nearest_neighbor,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from medmodels.medrecord.types import MedRecordAttributeInputList

Model: TypeAlias = Literal["logit", "dec_tree", "forest"]


def calculate_propensity(
    x_train: NDArray[Union[np.int64, np.float64]],
    y_train: NDArray[Union[np.int64, np.float64]],
    treated_test: NDArray[Union[np.int64, np.float64]],
    control_test: NDArray[Union[np.int64, np.float64]],
    model: Model = "logit",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculates the propensity/probabilities of a subject being in the treated group.

    A classification algorithm is trained, and it is used to predict the probability of
    a subject of being either in the treated or in the control set of patients.

    This function supports multiple classification algorithms and allows specifying
    hyperparameters. It is designed for binary classification tasks, focusing on the
    probability of the positive class.

    Args:
        x_train (NDArray[Union[np.int64, np.float64]]): Feature matrix for training.
        y_train (NDArray[Union[np.int64, np.float64]]): Target variable for training.
        treated_test (NDArray[Union[np.int64, np.float64]]): Feature matrix for the
            treated group to predict probabilities.
        control_test (NDArray[Union[np.int64, np.float64]]): Feature matrix for the
            control group to predict probabilities.
        model (Model, optional): Classification algorithm to use. Options: "logit",
            "dec_tree", "forest".
        hyperparameters (Optional[Dict[str, Any]], optional): Manual hyperparameter
            settings. Uses default hyperparameters if None.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]: Probabilities of the positive
            class for treated and control groups.

    Example:
        For "dec_tree" model with iris dataset inputs, returns probabilities of the
        last class for treated and control sets, e.g., ([0.], [0.]).
    """
    propensity_model = PROP_MODEL[model]
    pm = propensity_model(**hyperparameters) if hyperparameters else propensity_model()
    pm.fit(x_train, y_train)

    # Predict the probability of the treated and control groups
    treated_probability = np.array(pm.predict_proba(treated_test)).T[-1]
    control_probability = np.array(pm.predict_proba(control_test)).T[-1]

    return treated_probability, control_probability


def run_propensity_score(
    treated_set: pl.DataFrame,
    control_set: pl.DataFrame,
    model: Model = "logit",
    number_of_neighbors: int = 1,
    hyperparameters: Optional[Dict[str, Any]] = None,
    covariates: Optional[MedRecordAttributeInputList] = None,
) -> pl.DataFrame:
    """Executes Propensity Score matching using a specified classification algorithm.

    Constructs the training target by assigning 1 to the treated set and 0 to the
    control set, then predicts the propensity score. This score is used for matching
    using the nearest neighbor method.

    This function simplifies the process of propensity score matching, focusing on the
    use of the propensity score as the sole covariate for matching.

    Args:
        treated_set (pl.DataFrame): Data for the treated group.
        control_set (pl.DataFrame): Data for the control group.
        model (Model, optional): Classification algorithm for predicting probabilities.
            Options include "logit", "dec_tree", "forest".
        number_of_neighbors (int, optional): Number of nearest neighbors to find for
            each treated unit. Defaults to 1.
        hyperparameters (Optional[Dict[str, Any]], optional): Hyperparameters for model
            tuning. Increases computation time if set. Uses default if None.
        covariates (Optional[MedRecordAttributeInputList], optional): Features for
            matching. Uses all if None.

    Returns:
        pl.DataFrame: Matched subset from the control set corresponding to the treated
            set.
    """
    if not covariates:
        covariates = [col for col in treated_set.columns if col != "id"]

    treated_array = treated_set.select(covariates).to_numpy().astype(float)
    control_array = control_set.select(covariates).to_numpy().astype(float)

    x_train = np.concatenate((treated_array, control_array))
    len_treated, len_control = len(treated_array), len(control_array)
    y_train = np.concatenate((np.ones(len_treated), np.zeros(len_control)))

    treated_prop, control_prop = calculate_propensity(
        x_train,
        y_train,
        treated_array,
        control_array,
        hyperparameters=hyperparameters,
        model=model,
    )

    # Add propensity score to the original data
    treated_set = treated_set.with_columns(pl.Series("prop_score", treated_prop))
    control_set = control_set.with_columns(pl.Series("prop_score", control_prop))

    matched_control = nearest_neighbor(
        treated_set,
        control_set,
        number_of_neighbors=number_of_neighbors,
        covariates=["prop_score"],
    )

    matched_control = matched_control.drop("prop_score")
    treated_set = treated_set.drop("prop_score")
    control_set = control_set.drop("prop_score")

    return matched_control


PROP_MODEL = {
    "logit": LogisticRegression,
    "dec_tree": DecisionTreeClassifier,
    "forest": RandomForestClassifier,
}
