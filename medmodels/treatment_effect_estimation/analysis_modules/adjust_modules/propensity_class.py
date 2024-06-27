from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import polars as pl

from medmodels import MedRecord
from medmodels.matching.algorithms.classic_distance_models import nearest_neighbor
from medmodels.matching.algorithms.propensity_score import Model, calculate_propensity
from medmodels.matching.metrics import Metric
from medmodels.medrecord.types import MedRecordAttributeInputList, NodeIndex


class PropensityMatching:
    def __init__(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_group: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList = ["gender", "age"],
        one_hot_covariates: MedRecordAttributeInputList = ["gender"],
        model: Model = "logit",
        distance_metric: Metric = "absolute",
        number_of_neighbors: int = 1,
        hyperparam: Optional[Dict[str, Any]] = None,
    ):
        """Class for the propensity score matching.

        The algorithm trains the chosen classification method on the treated and control
        sets. Y_train is constructed as follows: 1 for each entry of the treated and 0
        for the control set. The probability of the class 1 will be assign as a new
        variable "Prop. score" and used for the nearest neighbor matching as the only
        covariate.

        Args:
            medrecord (MedRecord): medrecord object containing the data.
            treated_group (Set[NodeIndex]): Set of treated subjects.
            control_group (Set[NodeIndex]): Set of control subjects.
            essential_covariates (MedRecordAttributeInputList, optional): Covariates
                that are essential for matching. Defaults to ["gender", "age"].
            one_hot_covariates (MedRecordAttributeInputList, optional): Covariates that
                are one-hot encoded for matching. Defaults to ["gender"].
            model (Model, optional): classification method to be used, default: "logit".
                Can be chosen from ["logit", "dec_tree", "forest"].
            distance_metric (Metric, optional): metric to be used for the matching.
                Defaults to "absolute". Can be chosen from ["absolute", "exact",
                "mahalanobis"].
            number_of_neighbors (int, optional): number of neighbors to be matched per
                treated subject. Defaults to 1.
            hyperparam (Optional[Dict[str, Any]], optional): hyperparameters for the
                classification model, default: None.
        """
        self.essential_covariates = [
            str(covariate) for covariate in essential_covariates
        ]
        self.one_hot_covariates = [str(covariate) for covariate in one_hot_covariates]

        if "id" not in self.essential_covariates:
            self.essential_covariates.append("id")

        # Preprocess the data
        self.data_treated, self.data_control = self.preprocess_data(
            medrecord, treated_group, control_group
        )

        # Run the algorithm to find the matched controls
        matched_controls = self.run(
            model=model,
            number_of_neighbors=number_of_neighbors,
            distance_metric=distance_metric,
            hyperparam=hyperparam,
        )
        # Save the matched control ids
        self.matched_controls = set(matched_controls["id"])

    def preprocess_data(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_group: Set[NodeIndex],
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Preprocesses the data for the propensity score matching.

        The data is one-hot encoded and the essential covariates are selected. The
        treated and control sets are selected and the data is returned as polars
        dataframes.

        Args:
            medrecord (MedRecord): medrecord object containing the data.
            treated_group (Set[NodeIndex]): Set of treated subjects.
            control_group (Set[NodeIndex]): Set of control subjects.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: treated and control dataframes.
        """
        # Dataframe
        data = pl.DataFrame(
            data=[
                {"id": k, **v}
                for k, v in medrecord.node[list(control_group | treated_group)].items()
            ]
        )
        original_columns = data.columns

        # One-hot encode the categorical variables
        data = data.to_dummies(columns=self.one_hot_covariates, drop_first=True)
        new_columns = [col for col in data.columns if col not in original_columns]

        # Add to essential covariates the new columns created by one-hot encoding and
        # delete the ones that were one-hot encoded
        essential_covariates = self.essential_covariates.copy()
        essential_covariates.extend(new_columns)
        [essential_covariates.remove(col) for col in self.one_hot_covariates]
        data = data.select(essential_covariates)

        # Select the sets of treated and control subjects
        data_treated = data.filter(pl.col("id").is_in(treated_group))
        data_control = data.filter(pl.col("id").is_in(control_group))

        return data_treated, data_control

    def run(
        self,
        model: Model,
        number_of_neighbors: int,
        distance_metric: Metric,
        hyperparam: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        """
        Runs Propensity Score matching with the chosen classification method.

        The algorithm trains the chosen classification method on the treated and control
        sets. Y_train is constructed as follows: 1 for each entry of the treated and 0
        for the control set. The probability of the class 1 will be assign as a new
        variable "Prop. score" and used for the nearest neighbor matching as the only
        covariate.

        Args:
            model (Model): classification method to be used. Can be chosen from
                ["logit", "dec_tree", "forest"].
            number_of_neighbors (int): number of neighbors to be matched per treated
                subject.
            distance_metric (Metric): metric to be used for the matching. Can be chosen
                from ["absolute", "exact", "mahalanobis"].
            hyperparam (Optional[Dict[str, Any]], optional): hyperparameters for the
                classification model.

        Returns:
            pl.DataFrame: Matched subset from the control set.
        """
        logging.info("Running the propensity score matching algorithm")
        # We need to convert the data to float, but only the columns that can be
        # converted to float
        data_treated = pl.DataFrame()
        data_control = pl.DataFrame()

        # Convert columns to float where possible
        for col in self.data_treated.columns:
            try:
                data_treated = data_treated.with_columns(
                    self.data_treated[col].cast(pl.Float64, strict=True).alias(col)
                )
                data_control = data_control.with_columns(
                    self.data_control[col].cast(pl.Float64, strict=True).alias(col)
                )
            except (
                pl.exceptions.ComputeError
            ):  # If the column cannot be converted to float
                pass

        # Convert the Polars DataFrames to NumPy arrays
        treated_array = data_treated.to_numpy()
        control_array = data_control.to_numpy()

        # Train the classification model
        x_train = np.concatenate((treated_array, control_array))
        len_treated, len_control = len(treated_array), len(control_array)
        y_train = np.concatenate((np.ones(len_treated), np.zeros(len_control)))

        treated_prop, control_prop = calculate_propensity(
            x_train=x_train,
            y_train=y_train,
            treated_test=treated_array,
            control_test=control_array,
            hyperparam=hyperparam,
            model=model,
        )

        # Add propensity score to the original data polars dataframes
        self.data_treated = self.data_treated.with_columns(
            pl.Series("prop_score", treated_prop)
        )
        self.data_control = self.data_control.with_columns(
            pl.Series("prop_score", control_prop)
        )

        matched_control = nearest_neighbor(
            self.data_treated,
            self.data_control,
            number_of_neighbors=number_of_neighbors,
            metric=distance_metric,
            covariates=["prop_score"],
        )

        return matched_control.drop("prop_score")
