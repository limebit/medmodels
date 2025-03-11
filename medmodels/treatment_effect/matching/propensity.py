"""Module for the propensity score matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Set

import numpy as np
import polars as pl

from medmodels import MedRecord
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
    from medmodels.medrecord.types import Group, MedRecordAttribute, NodeIndex


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
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the propensity score class.

        Args:
            model (Model, optional): classification method to be used, default: "logit".
                Can be chosen from ["logit", "dec_tree", "forest"].
            number_of_neighbors (int, optional): number of neighbors to be matched per
                treated subject. Defaults to 1.
            hyperparameters (Optional[Dict[str, Any]], optional): hyperparameters for
                the classification model. Defaults to None.
        """
        super().__init__(number_of_neighbors)
        self.model = model
        self.number_of_neighbors = number_of_neighbors
        self.hyperparameters = hyperparameters

    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        patients_group: Group,
        essential_covariates: Optional[Sequence[MedRecordAttribute]] = None,
        one_hot_covariates: Optional[Sequence[MedRecordAttribute]] = None,
    ) -> Set[NodeIndex]:
        """Matches the controls based on propensity score matching.

        Args:
            medrecord (MedRecord): medrecord object containing the data.
            control_set (Set[NodeIndex]): Set of control subjects.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            patients_group (Group): Group of patients in MedRecord.
            essential_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are essential for matching. Defaults to None, meaning
                all the attributes of the patients are used.
            one_hot_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are one-hot encoded for matching. Defaults to None,
                meaning all the categorical attributes of the patients are used.

        Returns:
            Set[NodeIndex]:  Node Ids of the matched controls.
        """
        # Preprocess the data
        data_treated, data_control = self._preprocess_data(
            medrecord=medrecord,
            treated_set=treated_set,
            control_set=control_set,
            patients_group=patients_group,
            essential_covariates=list(essential_covariates)
            if essential_covariates
            else None,
            one_hot_covariates=list(one_hot_covariates) if one_hot_covariates else None,
        )

        # Convert the Polars DataFrames to NumPy arrays
        treated_array = data_treated.drop("id").cast(pl.Float64, strict=True).to_numpy()
        control_array = data_control.drop("id").cast(pl.Float64, strict=True).to_numpy()

        # Train the classification model
        x_train = np.concatenate((treated_array, control_array))
        y_train = np.concatenate(
            (
                np.ones(len(treated_array)),
                np.zeros(len(control_array)),
            )
        )

        # Calculate the propensity scores for the treated and control sets
        treated_propensity, control_propensity = calculate_propensity(
            x_train=x_train,
            y_train=y_train,
            treated_test=treated_array,
            control_test=control_array,
            hyperparameters=self.hyperparameters,
            model=self.model,
        )

        # Add propensity score to the original data polars dataframes
        data_treated = data_treated.with_columns(
            pl.Series("prop_score", treated_propensity)
        )
        data_control = data_control.with_columns(
            pl.Series("prop_score", control_propensity)
        )

        matched_control = nearest_neighbor(
            data_treated,
            data_control,
            number_of_neighbors=self.number_of_neighbors,
            covariates=["prop_score"],
        )

        return set(matched_control["id"])
