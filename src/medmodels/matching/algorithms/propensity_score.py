from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from medmodels.matching.algorithms.classic_distance_models import nearest_neighbor


def calculate_propensity(
    x_train, y_train, treated_test, control_test, hyperparam=None, metric="logit"
):

    """
    Trains one of the classification algorithms on x_train, y_train, predicts
    probability of the target variable for the treated_test and control_test
    data sets and returns the probability of the last class.

    Example (for "dec_tree" metric):
        x_trains, y_train: iris data set
        treated_test, control_test: [5.1, 3.5, 1.4, 0.2], [4.9, 3. , 1.4, 0.2]
        probabilities of the classes:
            treated: [1. 0. 0.]
            control: [1. 0. 0.]
        return: [0.], [0.]

    @param x_train: features of the data set;
    @param y_train: target feature of the data set;
    @param treated_test: test set for the treated proba;
    @param control_test: test set for the control proba;
    @param hyperparam: hyperparameter to set manually, if None the default parameters
                       will be taken;
    @param metric: the classification algorithm, can be chosen from:
                   - "logit": Logistische Regression;
                   - "dec_tree": Decision Tree Classifier;
                   - "forest": Random Forest Classifier.
    @return: arrays of probabilities of the last class for the both test sets
    """

    propensity_metric = PROP_MODEL[metric]

    if hyperparam:

        pm = propensity_metric(**hyperparam)

    else:

        pm = propensity_metric()

    pm.fit(x_train, y_train)

    return pm.predict_proba(treated_test).T[-1], pm.predict_proba(control_test).T[-1]


def run_propensity_score(
    treated_set,
    control_set,
    model="logit",
    hyperparam=None,
    covariates=None,
    log_text=None,
):

    """
    Runs Propensity Score matching with the chosen classification method.

    The algorithm trains the chosen classification method on the treated and control
    sets. Y_train is constructed as follows: 1 for each entry of the treated and 0 for
    the control set. The probability of the class 1 will be assign as a new variable
    "Prop. score" and used for the nearest neighbor matching as the only covariate.

    @param treated_set: set of treated items;
    @param control_set: set of untreated items;
    @param model: algorithm to predict the probabilities, can be chosen from
                  ["logit", "dec_tree", "forest"];
    @param hyperparam: number of variation for each hyper parameter to test with random
                        search method. This might increase the computation time;
    @param covariates: list of features to be considered for the matching, if None take
                       all of the variables;
    @param log_text: text for the progress bar, default: None;
    @return: dataframe as a subset of the control set matched for the treated set.
    """

    if not covariates:
        covariates = treated_set.columns

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
        treated_set,
        control_set,
        metric="absolute",
        covariates=["Prop. score"],
        log_text=log_text,
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
