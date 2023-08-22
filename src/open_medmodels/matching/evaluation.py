import pandas as pd


def relative_diff_in_means(control_set, treated_set):

    """
    Calculates the absolute relative mean for each covariate/feature as a difference
    between the control and treated in percent related to control.

    @param control_set: control set
    @param treated_set: treated set
    @return: dataframe with mean values of the control and treated set and absolute
             relative difference for all features
    """

    def calculate_relative_diff(row):

        """
        Calculates the absolute relative mean for one covariate as a difference between
        the control and treated in percent related to control.

        Let for a given feature x, y be its means over the treated and control set resp.
        then:
        .. math::
            Diff(x, y) = \begin{cases}
                         |x - y| & \text {if y = 0}
                         |\frac{x - y}{y}| & \text {otherwise}

        Example: control 2, treated 3.5 --> Diff 75 %
                 So, treatment changed this feature by 75 %.
                 Note, that treated 0.5 also returns Diff 75 %.
                 control 0, treated 0.2 --> Diff 20 %

        @param row: the feature to calculate the relative difference for;
        @return: the relative difference.
        """

        if row.control_mean == 0:
            return abs(row.treated_mean - row.control_mean) * 100
        else:
            return abs((row.treated_mean - row.control_mean) / row.control_mean) * 100

    control_mean = pd.DataFrame(control_set.mean()).transpose()
    treated_mean = pd.DataFrame(treated_set.mean()).transpose()
    df_mean = pd.concat([control_mean, treated_mean], ignore_index=True)

    df_mean = df_mean.rename(index={0: "control_mean", 1: "treated_mean"})

    df_mean = df_mean.transpose()

    df_mean["Diff (in %)"] = df_mean.apply(calculate_relative_diff, axis=1)

    return df_mean.transpose()


def average_value_over_features(df):

    """
    Returns the mean over the last row of a dataframe. This method can be used e.g. to
    calculate average difference between the treated and control sets.

    Example:                  a      b
              control_mean   2.0    2.0
              treated_mean   3.5    7.0   returns (75.0 + 250.0)/2 = 162.5
              Diff (in %)   75.0  250.0


    @param df: dataframe, the mean of the last row to be calculated for
    @return: mean value as float
    """

    return df.tail(1).mean(axis=1).values.tolist()[0]


def average_abs_relative_diff(control_set, treated_set, covariates=None):

    """
    Calculates the average absolute relative difference in means over all covariates.

    @param control_set: control set
    @param treated_set: treated set
    @param covariates: if not given consider all features as covariates
    @return: a mean value over all features as float, the dataframe with all means
    """

    if not covariates:
        covariates = treated_set.columns

    df_mean = relative_diff_in_means(control_set, treated_set)

    return average_value_over_features(df_mean[covariates]), df_mean
