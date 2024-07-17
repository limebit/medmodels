from medmodels.dataclass.dataclass import MedRecord
import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, List
from icdmappings import Mapper


def replace_ndc(df: pd.DataFrame, mode_ndc: dict) -> pd.DataFrame:
    """Replace NDC column values with the most repeated NDC value for each group.

    :param df: Dataframe with all prescriptions
    :type df: pd.DataFrame
    :param mode_ndc: Dictionary with the mode value for each drug
    :type mode_ndc: dict
    """
    drug = df["drug"].iloc[0]
    assert all(df["drug"] == drug)

    if drug in mode_ndc:
        mode_value = mode_ndc[drug]
        df["prescriptions_id"] = df["prescriptions_id"].map(
            lambda x: mode_value if x != mode_value else x
        )
    return df


def icd9toicd10(code: np.ndarray) -> List[str]:
    """Translate ICD9 codes to ICD10.

    :param code: ICD9 codes
    :type code: np.ndarray
    :return: ICD10 codes
    :rtype: List[str]
    """
    mapper = Mapper()
    return mapper.map(codes=code, mapper="icd9toicd10")


def medrecord_loader(medrecord: MedRecord, translate=False) -> pd.DataFrame:
    """Load a MedRecord object and adapt it to the necessary format for the
    MCE model.

    :param medrecord: MedRecord object
    :type medrecord: MedRecord
    :param translate: Whether to translate the ICD9 codes to ICD10. Defaults to False.
    :type translate: bool, optional
    return: Dataframe with the necessary format for the MCE model
    rtype: pd.DataFrame
    """
    med_df = medrecord.dimensions_to_dict()
    data = pd.DataFrame()

    for dim in medrecord.dimensions:
        if dim in ("patients", "admissions"):
            continue

        data_dim = med_df[dim]
        assert "time" in data_dim.columns, f"Time column not found in {dim} dimension"

        # First letter for defining the event
        first_letter = dim[0].upper()
        dim_id = dim + "_id"

        # Remove prefix from ids
        data_dim[dim_id] = data_dim[dim_id].str.replace(dim + "_", "", regex=False)

        if dim == "prescriptions":
            if translate:
                grouped = data_dim.groupby("drug", group_keys=True)

                # Calculate mode of ndc column for each group
                mode_ndc = grouped[dim_id].agg(lambda x: x.mode()[0]).to_dict()

                # Apply function to each group and concatenate DataFrames
                prescriptions = grouped.apply(replace_ndc, mode_ndc=mode_ndc)
            else:
                prescriptions = data_dim

            first_letter = "M"
            ndc_codes = prescriptions[dim_id].astype(float).astype(int).astype(str)
            data_dim["event"] = [f"{first_letter}-NDC-{code}" for code in ndc_codes]

        elif dim == "procedures":
            data_dim["event"] = (
                first_letter
                + "-ICD"
                + data_dim["icd_version"].astype(str)
                + "-"
                + data_dim[dim_id]
            )
        elif dim == "diagnoses":
            if translate:
                is_icd9 = data_dim["icd_version"].astype(int) == 9
                data_dim.loc[is_icd9, dim_id] = icd9toicd10(
                    data_dim.loc[is_icd9, dim_id].values
                )
                data_dim.loc[is_icd9, "icd_version"] = "10"

            data_dim["event"] = (
                first_letter
                + "-ICD"
                + data_dim["icd_version"].astype(str)
                + "-"
                + data_dim[dim_id]
            )
        else:
            data_dim["event"] = first_letter + "-" + data_dim[dim_id]

        # Convert time to datetime
        data_dim["time"] = pd.to_datetime(data_dim["time"], format="%Y-%m-%d %H:%M:%S")

        # Calculate relative time
        grouped = data_dim.groupby("patients_id")["time"]
        relative_time = (
            (data_dim["time"].values - grouped.transform("min").values)
            / np.timedelta64(1, "s")
            / 86400
        )
        data_dim["relative_time"] = relative_time.astype(float)

        # Only select the columns we need
        data_dim = data_dim.loc[
            :, ["patients_id", "event", "time", "relative_time"]
        ].copy()

        # Append to the list of dataframes
        data = pd.concat([data, data_dim], axis=0)

    data = data.dropna(axis=0)

    return data.sort_values(by=["patients_id", "time"]).reset_index(drop=True)


def find_clusters(
    point_cloud,
    num_clusters: int = 20,
    starting_distance: int = 5,
    distance_step: float = 0.25,
    min_cluster_size: int = 3,
    iterations: int = 50,
) -> Tuple[np.array, np.array]:
    """Find clusters in a point cloud using a Clique graph-based approach.

    :param point_cloud: Points
    :type point_cloud: np.array
    :param num_clusters: Number of clusters to find
    :type num_clusters: int
    :param starting_distance: Starting distance for the graph
    :type starting_distance: int
    :param distance_step: Distance step for the graph
    :type distance_step: float
    :param min_cluster_size: Minimum size for a cluster
    :type min_cluster_size: int
    :param iterations: Number of iterations to run the algorithm
    :type iterations: int
    :return: Clusters index for each point (0 meaning no cluster),
        and centers of each cluster.
    :rtype: Tuple[np.array, np.array]
    """
    assert min_cluster_size > 1, "Minimum cluster size must be greater than 1."

    obtained_clusters = 0
    distance = starting_distance

    # Insert a timeout timer for the while loop
    time = 0

    while num_clusters != obtained_clusters:
        # Timeout
        time += 1
        if time > iterations:
            print("Timeout, could not find the desired number of clusters.")
            break

        # Create graph
        graph = nx.Graph()
        for i in range(point_cloud.shape[0]):
            for j in range(i + 1, point_cloud.shape[0]):
                dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
                if dist < distance:
                    graph.add_edge(i, j)

        # Identify cliques with at least min_cluster_size elements
        cliques = [c for c in nx.find_cliques(graph) if len(c) >= min_cluster_size]

        # Merge overlapping cliques
        merged_cliques = []
        while cliques:
            c1 = cliques.pop()
            merged = False
            for i, c2 in enumerate(merged_cliques):
                if set(c1).intersection(set(c2)):
                    merged_cliques[i] = set(c1).union(set(c2))
                    merged = True
                    break
            if not merged:
                merged_cliques.append(set(c1))

        obtained_clusters = len(merged_cliques)
        if obtained_clusters > num_clusters or obtained_clusters == 0:
            distance += distance_step
        elif obtained_clusters < num_clusters:
            distance -= distance_step

    # Finding the center of mass of each cluster
    centers = []
    for c in merged_cliques:
        centers.append(np.mean(point_cloud[list(c)], axis=0))
    centers = np.array(centers)

    clusters = np.zeros(point_cloud.shape[0], dtype=int)
    for i, c in enumerate(merged_cliques):
        for j in c:
            clusters[j] = i + 1

    return clusters, centers
