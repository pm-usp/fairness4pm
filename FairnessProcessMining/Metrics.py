import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                          Fainess Metrics                                 """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def DisparateImpact(df, protected_attribute, privileged_group, unprivileged_group, prediction):
    """
    This function calculates Disparate Impact, a metric used to assess the disparity between 
    two groups (privileged and unprivileged) in the rate of positive outcomes of a predictive model. 
    Disparate Impact is calculated as the ratio of the rate of positive outcomes of the unprivileged 
    group to the rate of positive outcomes of the privileged group.
    The value returned by di provides a measure of fairness between the groups:
    - If di = 1, there is no disparity; the outcomes are equal between the groups.
    - If di < 1, the non-privileged group is at a disadvantage compared to the privileged group.
    - If di > 1, the non-privileged group is at an advantage compared to the privileged group.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g., race, gender).
    - privileged_group (any): Value representing the privileged group in the protected attribute.
    - unprivileged_group (any): Value representing the unprivileged group in the protected attribute.
    - prediction (str): Name of the column containing the model's predictions.
    
    Returns:
    - float: The Disparate Impact (ratio between the positive outcome rates of the unprivileged and privileged groups).
    """
    
    # Calculate the positive outcome rate for the privileged group
    privileged_positive_rate = df[df[protected_attribute] == privileged_group][prediction].mean()
    
    # Calculate the positive outcome rate for the unprivileged group
    unprivileged_positive_rate = df[df[protected_attribute] == unprivileged_group][prediction].mean()
    
    # Calculate DI
    di = unprivileged_positive_rate / privileged_positive_rate
    
    return di


def ModelDisparateImpact(df, protected_attribute, privileged_group, unprivileged_group, prediction):
    """
    This function calculates Disparate Impact, a metric used to assess the disparity between 
    two groups (privileged and unprivileged) in the rate of positive outcomes of a predictive model. 
    Disparate Impact is calculated as the ratio of the rate of positive outcomes of the unprivileged 
    group to the rate of positive outcomes of the privileged group.
    The value returned by di provides a measure of fairness between the groups:
    - If di = 1, there is no disparity; the outcomes are equal between the groups.
    - If di < 1, the non-privileged group is at a disadvantage compared to the privileged group.
    - If di > 1, the non-privileged group is at an advantage compared to the privileged group.
    
    Special Case Handling for Finite Return:
    - If both groups have a positive outcome rate of 0, or if one or both groups
    are empty in such a way that the rates are either undefined or both are zero, DI = 1.0.
    - If the rate of the advantaged group is 0 and that of the unprivileged group > 0,
    DI = 100.0 (representing a large disparity, unfavorable to the advantaged group).
    - If the rate of the unprivileged group is 0 (and that of the advantaged group > 0), DI = 0.0.
    - If one group is empty (resulting in a NaN rate) and the other has a rate > 0,
    DI = 0.0 (representing a computation difficulty or maximum apparent disparity).
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g., race, gender).
    - privileged_group (any): Value representing the privileged group in the protected attribute.
    - unprivileged_group (any): Value representing the unprivileged group in the protected attribute.
    - prediction (str): Name of the column containing the model's predictions.
    
    Returns:
    - float: The Disparate Impact (ratio between the positive outcome rates of the unprivileged and privileged groups).
    """
    
    df_privileged = df[df[protected_attribute] == privileged_group]
    df_unprivileged = df[df[protected_attribute] == unprivileged_group]

    # Calculates rates, .mean() on empty Series returns NaN
    if df_privileged.empty:
        privileged_positive_rate = np.nan
    else:
        # Ensures the prediction column is numeric for .mean()
        privileged_positive_rate = pd.to_numeric(df_privileged[prediction], errors='coerce').mean()

    if df_unprivileged.empty:
        unprivileged_positive_rate = np.nan
    else:
        unprivileged_positive_rate = pd.to_numeric(df_unprivileged[prediction], errors='coerce').mean()

    # Scenario 1: Both rates are NaN (ex: both groups empty)
    if np.isnan(privileged_positive_rate) and np.isnan(unprivileged_positive_rate):
        return 1.0

    # Scenario 2: Privileged group rate is NaN, but non-privileged rate is not.
    if np.isnan(privileged_positive_rate):
        # If the unprivileged rate is also 0 (or became NaN and then 0), DI = 1.0
        # If the unprivileged rate > 0, it is a "bad" situation for comparison.
        return 1.0 if unprivileged_positive_rate == 0 or np.isnan(unprivileged_positive_rate) else 0.0

    # Scenario 3: Non-privileged group rate is NaN, but privileged group rate is not.
    if np.isnan(unprivileged_positive_rate):
        # If the privileged rate is also 0 (or became NaN and then 0), DI = 1.0
        # If the privileged rate > 0, it is a "bad" situation.
        return 1.0 if privileged_positive_rate == 0 or np.isnan(privileged_positive_rate) else 0.0

    # Scenario 4: Privileged group rate is 0.
    if privileged_positive_rate == 0:
        if unprivileged_positive_rate == 0:
            # Both rates are 0. Considered no disparity.
            return 1.0
        else:  # unprivileged_positive_rate > 0
            # "Infinite" disparity in favor of the underprivileged group.
            # Returns a large, finite number to penalize in the fairness score.
            return 100.0 
    
    # Scenario 5: Privileged group rate > 0 (default case for split)
    # The non-privileged rate can be 0 here.
    di = unprivileged_positive_rate / privileged_positive_rate
    
    # Final safety: if 'di' is still NaN or Inf (it shouldn't with the logic above)
    if np.isnan(di) or np.isinf(di):
        return 1.0 # Safe and neutral fallback
        
    return di


def EqualityOfOpportunity(df, protected_attribute, target, prediction):
    """
    This function calculates equality of opportunity by evaluating the difference between 
    the true positive rates (TPR) for different groups (based on a protected attribute). 
    Equality of opportunity is achieved when the TPR is equal for all groups. 
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g., race, gender).
    - target (str): Name of the column containing the actual (true) values ​​of the outcome.  
    - prediction (str): Name of the column containing the model's predictions.
    
    Returns: 
    - pd.Series: The true positive rates (TPR) for each group. 
    - float: The maximum difference between the groups' TPRs, indicating disparity.
    """
    
    # Instances where both the actual and prediction values ​​are 1 (true positives)
    true_positive = df[(df[target] == 1) & (df[prediction] == 1)]
    
    # Instances where the actual value is 1, regardless of the prediction
    condition_positive = df[df[target] == 1]
    
    # TPR: proportion of true positives over all positive conditions
    tpr = (true_positive[protected_attribute].value_counts() / 
           condition_positive[protected_attribute].value_counts()).fillna(0)
    
    # Maximum difference between the TPRs of the groups
    max_diff_tpr = tpr.max() - tpr.min()
    
    return tpr, max_diff_tpr


def EqualizedOdds(df, protected_attribute, target, prediction):
    """
    This function calculates the True Positive Rate (TPR) and False Positive Rate (FPR) for 
    different groups (based on a protected attribute) and checks for disparity between these rates. 
    Equalized odds are achieved when the TPRs and FPRs are equal for all groups.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g., race, gender).
    - target (str): Name of the column containing the actual (true) values ​​of the outcome.  
    - prediction (str): Name of the column containing the model's predictions.

    Returns:
    - pd.Series: The true positive rates (TPR) for each group.
    - float: The maximum difference between the TPRs of the groups.
    - pd.Series: The false positive rates (FPR) for each group.
    - float: The maximum difference between the FPRs of the groups.
    """
    
    # Identification of true and false positives
    true_positive = df[(df[target] == 1) & (df[prediction] == 1)]
    false_positive = df[(df[target] == 0) & (df[prediction] == 1)]
    
    # Identification of positive and negative conditions
    condition_positive = df[df[target] == 1]
    condition_negative = df[df[target] == 0]
    
    # TPR: the proportion of true positives to total positive conditions
    tpr = (true_positive[protected_attribute].value_counts() / 
           condition_positive[protected_attribute].value_counts()).fillna(0)
    
    # FPR: the proportion of false positives to total negative conditions
    fpr = (false_positive[protected_attribute].value_counts() / 
           condition_negative[protected_attribute].value_counts()).fillna(0)
    
    # Calculation of the largest differences between rates for groups
    max_diff_tpr = tpr.max() - tpr.min()
    max_diff_fpr = fpr.max() - fpr.min()

    return tpr, max_diff_tpr, fpr, max_diff_fpr


def EqualizedOddsRatio(df, protected_attribute, target, prediction):
    """
    This function calculates the true positive rates (TPR) and false positive rates (FPR) 
    for different groups (based on a protected attribute) and checks for disparity between these rates.
    Now, disparity is measured by the ratio (max/min) instead of the difference (max - min).
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g., race, gender).
    - target (str): Name of the column containing the actual (true) values ​​of the outcome.  
    - prediction (str): Name of the column containing the model's predictions.

    Returns:
    - pd.Series: The true positive rates (TPR) for each group.
    - float: The maximum ratio of the groups' TPRs (TPR_max / TPR_min).
    - pd.Series: The false positive rates (FPR) for each group.
    - float: The maximum ratio of the groups' FPRs (FPR_max / FPR_min).
    """

    # Identification of true and false positives
    true_positive = df[(df[target] == 1) & (df[prediction] == 1)]
    false_positive = df[(df[target] == 0) & (df[prediction] == 1)]

    # Identification of positive and negative conditions
    condition_positive = df[df[target] == 1]
    condition_negative = df[df[target] == 0]

    # TPR: the proportion of true positives to total positive conditions
    tpr = (
        true_positive[protected_attribute].value_counts(dropna=False) 
        / condition_positive[protected_attribute].value_counts(dropna=False)
    ).fillna(0)

    # FPR: the proportion of false positives to total negative conditions
    fpr = (
        false_positive[protected_attribute].value_counts(dropna=False) 
        / condition_negative[protected_attribute].value_counts(dropna=False)
    ).fillna(0)

    # Calculation of the ratio (max/min) between the rates for the groups
    def ratio_max_min(series):
        min_val = series.min()
        max_val = series.max()
        if min_val == 0 and max_val == 0:
            return 1.0  # All groups have 0 => ratio 1 (no disparity)
        elif min_val == 0:
            return float('inf')  # If the smallest value is 0 and the largest > 0 => infinite ratio
        else:
            return max_val / min_val

    ratio_tpr = ratio_max_min(tpr)
    ratio_fpr = ratio_max_min(fpr)

    return tpr, ratio_tpr, fpr, ratio_fpr



def DemographicParity(df, protected_attribute, prediction):
    """
    This function calculates the Demographic Parity between different groups defined by a 
    protected attribute. In other words, it evaluates whether the positive prediction rate 
    (p(Y_pred=1)) is similar between groups.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g., race, gender).
    - prediction (str): Name of the column containing the model's predictions.

    Returns:
    - pd.Series: The positive prediction rates (PredPositiveRate) for each group (index = groups).
    - float: The maximum difference between the positive prediction rates of the groups, indicating disparity.
    """
    
    # Select only those instances where the prediction was positive
    predicted_positive = df[df[prediction] == 1]
    
    # Positive prediction rate by protected group:
        # (Number of cases predicted as 1 in the group) / (total number of cases in the group)
        # pos_rate: This is a Series with the positive prediction rate for each value (or group) of the protected attribute.
    pos_rate = (predicted_positive[protected_attribute].value_counts() 
                / df[protected_attribute].value_counts()).fillna(0)
    
    # Maximum difference between the positive prediction rates of the groups
    # It is the difference between the highest and lowest rate among all the values ​​of the pos_rate series
    max_diff_dp = pos_rate.max() - pos_rate.min()
    
    return pos_rate, max_diff_dp


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                Dataframe com o cálculo das métricas                      """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ModelMetrics(df_final, protected_attribute, dataset, model, privileged_group, unprivileged_group):
    """
    This function calculates a series of performance and fairness metrics for a machine learning model,
    for both the training and test sets. The metrics include accuracy, precision, recall, F1-Score, AUC,
    Disparate Impact, Equality of Opportunity (TPR), Equalized Odds (TPR and FPR) and their respective maximum differences.
    The results are organized in a DataFrame that contains the metrics, their values, the source (training or testing),
    the model type and the dataset.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing the predictions and the 'source' column indicating 'train' or 'test'.
    - protected_attribute (str): Name of the protected attribute column.
    - dataset (str): Name of the dataset (to identify in the output DataFrame).
    - model (str): Name of the model (to identify in the output DataFrame). - privileged_group (int/float): Value representing the privileged group in the protected attribute (default: 0).
    - unprivileged_group (int/float): Value representing the unprivileged group in the protected attribute (default: 1).
    
    Returns:
    - df_results (pd.DataFrame): DataFrame containing all calculated metrics, including source (train/test),
    model type, and dataset name.
    """
    
    # Separate training and testing data
    df_train = df_final[df_final['source'] == 'train']
    df_test = df_final[df_final['source'] == 'test']

    # Calculates fairness metrics
    di_train = ModelDisparateImpact(df_train, protected_attribute, privileged_group, unprivileged_group, 'prediction')
    di_test = ModelDisparateImpact(df_test, protected_attribute, privileged_group, unprivileged_group, 'prediction')
    eop_train_tpr, eop_train_max_tpr = EqualityOfOpportunity(df_train, protected_attribute, 'Target', 'prediction')
    eop_test_tpr, eop_test_max_tpr = EqualityOfOpportunity(df_test, protected_attribute, 'Target', 'prediction')
    eo_train_tpr, eo_train_max_tpr, eo_train_fpr, eo_train_max_fpr = EqualizedOdds(df_train, protected_attribute, 'Target', 'prediction')
    eo_test_tpr, eo_test_max_tpr, eo_test_fpr, eo_test_max_fpr = EqualizedOdds(df_test, protected_attribute, 'Target', 'prediction')
    pos_rate_train, max_diff_dp_train = DemographicParity(df_train, protected_attribute, 'prediction')
    pos_rate_test, max_diff_dp_test = DemographicParity(df_test, protected_attribute, 'prediction')
    
    
    # Prepare metrics for training and testing
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", 
               "Disparate Impact", 
               "Equality of Opportunity (TPR)", "Max Difference - Equality of Opportunity (TPR)",
               "Equalized Odds (TPR)", "Max Difference - Equalized Odds (TPR)",
               "Equalized Odds (FPR)", "Max Difference - Equalized Odds (FPR)",
               "Demographic Parity (Rate)", "Max Difference - Demographic Parity"]
    train_values = [
        accuracy_score(df_train['Target'], df_train['prediction']),
        precision_score(df_train['Target'], df_train['prediction'], zero_division=0),
        recall_score(df_train['Target'], df_train['prediction'], zero_division=0),
        f1_score(df_train['Target'], df_train['prediction'], average='weighted'),
        roc_auc_score(df_train['Target'], df_train['prediction']),
        di_train,
        eop_train_tpr,
        eop_train_max_tpr,
        eo_train_tpr,
        eo_train_max_tpr,
        eo_train_fpr,
        eo_train_max_fpr,
        pos_rate_train, 
        max_diff_dp_train
    ]
    test_values = [
        accuracy_score(df_test['Target'], df_test['prediction']),
        precision_score(df_test['Target'], df_test['prediction'], zero_division=0),
        recall_score(df_test['Target'], df_test['prediction'], zero_division=0),
        f1_score(df_test['Target'], df_test['prediction'], average='weighted'),
        roc_auc_score(df_test['Target'], df_test['prediction']),
        di_test,
        eop_test_tpr,
        eop_test_max_tpr,
        eo_test_tpr,
        eo_test_max_tpr,
        eo_test_fpr,
        eo_test_max_fpr,
        pos_rate_test, 
        max_diff_dp_test
    ]

    # Creating the results DataFrame
    extended_metrics = metrics * 2  # Duplicate the list of metrics
    values = train_values + test_values  # Concatenate training and test values
    sources = ["Train"] * len(metrics) + ["Test"] * len(metrics)  # Identify the source of each value
    types = [model] * len(extended_metrics)  # Model type for all inputs

    df_results = pd.DataFrame({
        "Metric": extended_metrics,
        "Value": values,
        "Source": sources,
        "Type": types, 
        "Dataset": dataset
    })

    return df_results
