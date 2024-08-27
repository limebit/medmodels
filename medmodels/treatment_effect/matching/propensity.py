from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import numpy as np
import polars as pl

from medmodels.treatment_effect.matching.algorithms.classic_distance_models import (
    nearest_neighbor,
)
from medmodels.treatment_effect.matching.algorithms.propensity_score import (
    Model,
    calculate_propensity,
)
from medmodels.treatment_effect.matching.matching import Matching

if TYPE_CHECKING:
    from medmodels import MedRecord
    from medmodels.medrecord.types import MedRecordAttributeInputList, NodeIndex


class PropensityMatching(Matching):
    """Class for the propensity score matching.

    The algorithm trains the chosen classification method on the treated and control
    sets. Y_train is constructed as follows: 1 for each entry of the treated and 0
    for the control set. The probability of the class 1 will be assign as a new
    variable "Prop. score" and used for the nearest neighbor matching as the only
    covariate.
    """

    model: Model
    number_of_neighbors: int
    hyperparam: Optional[Dict[str, Any]]

    def __init__(
        self,
        *,
        model: Model = "logit",
        number_of_neighbors: int = 1,
        hyperparam: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the propensity score class.

        Args:
            model (Model, optional): classification method to be used, default: "logit".
                Can be chosen from ["logit", "dec_tree", "forest"].
            nearest_neighbors_algorithm (NNAlgorithm, optional): algorithm used to
                compute nearest neighbors. Defaults to "auto".
            number_of_neighbors (int, optional): number of neighbors to be matched per
                treated subject. Defaults to 1.
            hyperparam (Optional[Dict[str, Any]], optional): hyperparameters for the
                classification model, default: None.
        """
        self.model = model
        self.number_of_neighbors = number_of_neighbors
        self.hyperparam = hyperparam

    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_group: Set[NodeIndex],
        treated_group: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList = None,
        one_hot_covariates: MedRecordAttributeInputList = None,
    ) -> Set[NodeIndex]:
        """Matches the controls based on propensity score matching.

        Args:
            medrecord (MedRecord): medrecord object containing the data.
            treated_group (Set[NodeIndex]): Set of treated subjects.
            control_group (Set[NodeIndex]): Set of control subjects.
            essential_covariates (MedRecordAttributeInputList, optional): Covariates
                that are essential for matching. Defaults to ["gender", "age"].
            one_hot_covariates (MedRecordAttributeInputList, optional): Covariates that
                are one-hot encoded for matching. Defaults to ["gender"].

        Returns:
            Set[NodeIndex]:  Node Ids of the matched controls.
        """
        # Preprocess the data
        if one_hot_covariates is None:
            one_hot_covariates = ["gender"]
        if essential_covariates is None:
            essential_covariates = ["gender", "age"]
        data_treated, data_control = self._preprocess_data(
            medrecord=medrecord,
            treated_group=treated_group,
            control_group=control_group,
            essential_covariates=essential_covariates,
            one_hot_covariates=one_hot_covariates,
        )
        # Convert the Polars DataFrames to NumPy arrays
        treated_array = data_treated.drop("id").cast(pl.Float64, strict=True).to_numpy()
        control_array = data_control.drop("id").cast(pl.Float64, strict=True).to_numpy()

        # Train the classification model
        x_train = np.concatenate((treated_array, control_array))
        y_train = np.concatenate(
            (np.ones(len(treated_array)), np.zeros(len(control_array)))
        )

        treated_prop, control_prop = calculate_propensity(
            x_train=x_train,
            y_train=y_train,
            treated_test=treated_array,
            control_test=control_array,
            hyperparam=self.hyperparam,
            model=self.model,
        )

        # Add propensity score to the original data polars dataframes
        data_treated = data_treated.with_columns(pl.Series("prop_score", treated_prop))
        data_control = data_control.with_columns(pl.Series("prop_score", control_prop))

        matched_control = nearest_neighbor(
            data_treated,
            data_control,
            number_of_neighbors=self.number_of_neighbors,
            covariates=["prop_score"],
        )

        return set(matched_control["id"])
