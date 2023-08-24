import pandas as pd
from medmodels.matching import metrics
import numpy as np


def nearest_neighbor(treated_set, control_set, metric, log_text=None, covariates=None):

    """
    Matches two dataframes with the Nearest Neighbor (greedy) algorithm. It runs through
    the treated set and selects its closest element from the control set. Each choice of
    unit occurs without reference to other pairings (greedy algorithm), and therefore
    does not aim to optimize the matching in general. Nearest neighbor is the most
    common algorithm for matching and can be used with different metrics.

    Usually treated set is smaller than control set. So, for each element of the treated
    set, elements from the control set are chosen. Indeed, sometimes control sets are
    smaller than treated sets and the matching will be done vice versa. For this reason
    a check should be done before executing the method and the variables renamed.

    @param treated_set: the dataframe to do the matching for;
    @param control_set: the dataframe to pick elements from;
    @param metric: the name of the metric (needs to be specified to define the
                   "closeness" between units. Can be chosen from
                   ["absolute", "mahalanobis"];
    @param log_text: text for the progress bar, default: None;
    @param covariates: features that are considered for the matching, if None all the
                       variables will be considered;
    @return: dataframe as a subset of the control set matched for the treated set.
    """

    metric_function = metrics.METRICS[metric]
    columns = treated_set.columns

    if not covariates:
        covariates = columns

    treated_array = treated_set[covariates].to_numpy().astype(float)
    control_array = control_set[covariates].to_numpy().astype(float)
    control_array_full = control_set.to_numpy()  # To keep all the infos
    matched_group = pd.DataFrame(columns=columns)

    for element_ss in treated_array:

        dist = []

        if metric == "mahalanobis":
            # Calculate the covariance matrix
            cov = np.cov(np.concatenate((treated_array, control_array)).T)
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.array([1 / cov])  # For the 1D case

            for element_bs in control_array:
                dist.append(metric_function(element_ss, element_bs, inv_cov=inv_cov))
        else:

            for element_bs in control_array:
                dist.append(metric_function(element_ss, element_bs))

        nn_index = np.argmin(dist)

        new_row = pd.DataFrame(control_array_full[nn_index], index=columns)
        matched_group = pd.concat([matched_group, new_row.transpose().astype(float)])
        # For the k:1 matching don't consider the chosen row any more.
        control_array_full = np.delete(control_array_full, nn_index, 0)
        control_array = np.delete(control_array, nn_index, 0)

    return matched_group.reset_index(drop=True)


ALGORITHMS = {"nearest neighbor": nearest_neighbor}
