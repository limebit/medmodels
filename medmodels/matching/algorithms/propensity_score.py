from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from medmodels.matching.algorithms.classic_distance_models import nearest_neighbor

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias


Model: TypeAlias = Literal["logit", "dec_tree", "forest"]


def calculate_propensity(
    x_train: NDArray[Union[np.int64, np.float64]],
    y_train: NDArray[Union[np.int64, np.float64]],
    treated_test: NDArray[Union[np.int64, np.float64]],
    control_test: NDArray[Union[np.int64, np.float64]],
    hyperparam: Optional[Dict[str, Any]] = None,
    metric: str = "logit",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Trains a classification algorithm on training data, predicts the probability of
    being in the last class for treated and control test datasets, and returns these
    probabilities.

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
        hyperparam (Optional[Dict[str, Any]], optional): Manual hyperparameter settings.
            Uses default if None.
        metric (str, optional): Classification algorithm to use. Options: "logit",
            "dec_tree", "forest".

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]: Probabilities of the positive
            class for treated and control groups.

    Example:
        For "dec_tree" metric with iris dataset inputs, returns probabilities of the
        last class for treated and control sets, e.g., ([0.], [0.]).
    """

    propensity_model = PROP_MODEL[model]

    if hyperparam:
        pm = propensity_model(**hyperparam)

    else:
        pm = propensity_model()

    pm.fit(x_train, y_train)

    # Predict the probability of the treated and control groups
    treated_probability = np.array(pm.predict_proba(treated_test)).T[-1]
    control_probability = np.array(pm.predict_proba(control_test)).T[-1]

    return treated_probability, control_probability


def run_propensity_score(
    treated_set: pd.DataFrame,
    control_set: pd.DataFrame,
    model: str = "logit",
    hyperparam: Optional[Any] = None,
    covariates: Optional[Union[List[str], pd.Index[str]]] = None,
) -> pd.DataFrame:
    """
    Executes Propensity Score matching using a specified classification algorithm.
    Constructs the training target by assigning 1 to the treated set and 0 to the
    control set, then predicts the propensity score. This score is used for matching
    using the nearest neighbor method.

    Args:
        treated_set (pd.DataFrame): Data for the treated group.
        control_set (pd.DataFrame): Data for the control group.
        model (str, optional): Classification algorithm for predicting probabilities.
            Options include "logit", "dec_tree", "forest".
        hyperparam (Optional[Any], optional): Hyperparameters for model tuning.
            Increases computation time if set.
        covariates (Optional[Union[List[str], pd.Index[str]]], optional): Features for
            matching. Uses all if None.

    Returns:
        pd.DataFrame: Matched subset from the control set corresponding to the treated
            set.

    This function simplifies the process of propensity score matching, focusing on the
    use of the propensity score as the sole covariate for matching.
    """

    if not covariates:
        covariates = [col for col in treated_set.columns if col != "id"]

    treated_array = treated_set[covariates].to_numpy().astype(float)
    control_array = control_set[covariates].to_numpy().astype(float)

    x_train = np.concatenate((treated_array, control_array))
    len_treated, len_control = len(treated_array), len(control_array)
    y_train = np.concatenate((np.ones(len_treated), np.zeros(len_control)))

    treated_prop, control_prop = calculate_propensity(
        x_train,
        y_train,
        treated_array,
        control_array,
        hyperparam=hyperparam,
        metric=model,
    )

    # Add propensity score to the original data
    treated_set["Prop. score"] = treated_prop
    control_set["Prop. score"] = control_prop

    matched_control = nearest_neighbor(
        treated_set, control_set, metric="absolute", covariates=["Prop. score"]
    )

    matched_control.pop("Prop. score")
    treated_set.pop("Prop. score")
    control_set.pop("Prop. score")

    return matched_control


PROP_MODEL = {
    "logit": LogisticRegression,
    "dec_tree": DecisionTreeClassifier,
    "forest": RandomForestClassifier,
}
