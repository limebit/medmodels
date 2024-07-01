from __future__ import annotations

from typing import Set, Tuple

import polars as pl

from medmodels import MedRecord
from medmodels.matching.metrics import Metric
from medmodels.matching.algorithms.classic_distance_models import nearest_neighbor
from medmodels.medrecord.types import (
    MedRecordAttributeInputList,
    NodeIndex,
)


class NeighborsMatching:
    def __init__(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_group: Set[NodeIndex],
        essential_covariates: MedRecordAttributeInputList = ["gender", "age"],
        one_hot_covariates: MedRecordAttributeInputList = ["gender"],
        distance_metric: Metric = "absolute",
        number_of_neighbors: int = 1,
    ):
        """Class for the nearest neighbor matching.

        The algorithm finds the nearest neighbors in the control group for each treated
        subject based on the given distance metric. The essential covariates are used
        for matching, and the one-hot covariates are one-hot encoded. The matched
        control subjects are saved in the matched_controls attribute.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            treated_group (Set[NodeIndex]): Set of treated subjects.
            control_group (Set[NodeIndex]): Set of control subjects.
            essential_covariates (MedRecordAttributeInputList, optional): Covariates
                that are essential for matching
            one_hot_covariates (MedRecordAttributeInputList, optional): Covariates that
                are one-hot encoded for matching
            distance_metric (Metric, optional): Metric for matching. Defaults to
                "absolute".
            number_of_neighbors (int, optional): Number of nearest neighbors to find for
                each treated unit. Defaults to 1.
        """
        self.essential_covariates = [
            str(covariate) for covariate in essential_covariates
        ]
        self.one_hot_covariates = [str(covariate) for covariate in one_hot_covariates]

        if "id" not in self.essential_covariates:
            self.essential_covariates.append("id")

        # Preprocess the data
        self.data_treated, self.data_control = self.preprocess_data(
            medrecord, control_group, treated_group
        )

        # Run the algorithm to find the matched controls
        matched_controls = self.run(
            number_of_neighbors=number_of_neighbors,
            distance_metric=distance_metric,
        )
        # Save the matched control ids
        self.matched_controls = set(matched_controls["id"])

    def preprocess_data(
        self,
        medrecord: MedRecord,
        control_group: Set[NodeIndex],
        treated_group: Set[NodeIndex],
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Preprocesses the data for the propensity score matching.

        The data is one-hot encoded and the essential covariates are selected. The
        treated and control sets are selected and the data is returned as polars
        dataframes.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            control_group (Set[NodeIndex]): Set of control subjects.
            treated_group (Set[NodeIndex]): Set of treated subjects.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Dataframes for the treated and control
                groups.
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
        number_of_neighbors: int,
        distance_metric: Metric,
    ) -> pl.DataFrame:
        """Runs the nearest neighbors algorithm to find the matched controls.

        Args:
            number_of_neighbors (int): Number of nearest neighbors to find for each
                treated unit.
            distance_metric (Metric): Metric for matching.

        Returns:
            pl.DataFrame: Matched subset from the control set.
        """
        matched_controls = nearest_neighbor(
            self.data_treated,
            self.data_control,
            number_of_neighbors=number_of_neighbors,
            metric=distance_metric,
        )

        return matched_controls
