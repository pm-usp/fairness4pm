# Libraries for data manipulation
import numpy as np
import random
from typing import Optional, List, Dict, Any

# Libraries for fairness
from aif360.datasets import BinaryLabelDataset, StandardDataset

# Library for hyperparameter optimization
import optuna

#Sementes
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                          Auxiliary function                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute):
    """
    This function creates the training and testing datasets in the format compatible with the AIF360 library.
    
    Parameters:
    - X_train (pd.DataFrame): DataFrame with the features of the training set.
    - y_train (pd.Series): Series containing the labels of the training set.
    - X_test (pd.DataFrame): DataFrame with the features of the test set.
    - y_test (pd.Series): Series containing the labels of the test set.
    - protected_attribute (str): Name of the column containing the protected attribute (e.g. gender, race).
    
    Returns:
    - dataset_train (BinaryLabelDataset): Training dataset formatted for AIF360.
    - dataset_test (BinaryLabelDataset): Test dataset formatted for AIF360.
    """
    
    df_for_aif360_train = X_train.copy()
    df_for_aif360_train[y_train.name] = y_train 
    
    dataset_train = StandardDataset(df=df_for_aif360_train,
                                    label_name=y_train.name, 
                                    favorable_classes=[1],     
                                    protected_attribute_names=[protected_attribute],
                                    privileged_classes=[[0]]) 
                                   
    df_for_aif360_test = X_test.copy()
    df_for_aif360_test[y_test.name] = y_test 

    dataset_test = StandardDataset(df=df_for_aif360_test,
                                   label_name=y_test.name,
                                   favorable_classes=[1],
                                   protected_attribute_names=[protected_attribute],
                                   privileged_classes=[[0]])
    
    return dataset_train, dataset_test



def prepare_aif360_data(X_df_features, y_series_labels, protected_attribute_name):
    """
    Prepares input data in the AIF360 StandardDataset format.
    """
    df_combined = X_df_features.copy()
    # Ensure that y_series_labels has the correct name
    current_label_name = y_series_labels.name if y_series_labels.name is not None else 'Target'
    df_combined[current_label_name] = y_series_labels
    
    return StandardDataset(df_combined, 
                           label_name=current_label_name, 
                           favorable_classes=[1], 
                           protected_attribute_names=[protected_attribute_name], 
                           privileged_classes=[[0]])



def select_best_hyperparameters_trial(
    study: optuna.study.Study,
    di_tier1_min: float = 0.8,
    di_tier1_max: float = 1.25,
    di_tier2_min: float = 0.0,
    di_tier2_max: float = 1.0):
    
    """
    This function selects the best trial from an Optuna study based on a custom logic for 
    F1-score and Disparate Impact (DI).
    
    The selection logic is hierarchical:
    
    1. Tier 1: Prioritizes trials with DI within the range [di_tier1_min, di_tier1_max], 
    selecting the one with the highest F1-score.
    
    2. Tier 2 (Fallback): If no trial meets Tier 1, searches for trials with DI in the 
    range [di_tier2_min, di_tier2_max (exclusive)), sorting first by DI (descending, to the closest to di_tier2_max) 
    and then by F1-score (descending) as a tiebreaker.
    
    3. Overall Fallback: If no trial meets Tier 1 or Tier 2, selects the trial with the highest overall 
    F1-score among all valid trials.

    Parameters:
    - study: The optuna.study.Study object after executing study.optimize().
    - di_tier1_min: Lower bound of the DI for Tier 1 Criterion.
    - di_tier1_max: Upper bound of the DI for Tier 1 Criterion.
    - di_tier2_min: Lower bound of the DI for Tier 2 Criterion.
    - di_tier2_max: Upper bound (exclusive) of the DI for Tier 2 Criterion.

    Returns:
    - An optuna.trial.FrozenTrial object representing the best selected trial, or None if no suitable trial was found.
    """

    all_completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not all_completed_trials:
        print("LOG:select_best_trial: Error - No Optuna trials have been completed.")
        return None

    # Prepare a list of dictionaries to facilitate sorting and access
    # Consider only trials with valid 'avg_f1' and 'avg_di' metrics and finite DI
    valid_trials_info: List[Dict[str, Any]] = []
    for t in all_completed_trials:
        f1 = t.user_attrs.get("avg_f1")
        di = t.user_attrs.get("avg_di")
        if f1 is not None and di is not None and np.isfinite(di):
            valid_trials_info.append({"trial_obj": t, "f1": f1, "di": di})

    if not valid_trials_info:
        print(
            "LOG:select_best_trial: Error - No completed trial has valid 'avg_f1' and 'avg_di'"
        )
        return None

    selected_trial_obj: Optional[optuna.trial.FrozenTrial] = None

    # Tier 1 criteria
    tier1_candidates = [
        info for info in valid_trials_info if di_tier1_min <= info["di"] <= di_tier1_max
    ]
    if tier1_candidates:
        tier1_candidates.sort(key=lambda x: x["f1"], reverse=True)
        selected_trial_obj = tier1_candidates[0]["trial_obj"]
        print(
            f"LOG:select_best_trial: Selected via Tier 1 criteria"
            f"(DI in [{di_tier1_min:.2f}, {di_tier1_max:.2f}], highest F1). "
            f"F1: {tier1_candidates[0]['f1']:.4f}, DI: {tier1_candidates[0]['di']:.4f}"
        )
    else:
        # Tier 2 criteria
        tier2_candidates = [
            info for info in valid_trials_info if di_tier2_min <= info["di"] < di_tier2_max
        ]
        if tier2_candidates:
            tier2_candidates.sort(key=lambda x: (x["di"], x["f1"]), reverse=True)
            selected_trial_obj = tier2_candidates[0]["trial_obj"]
            print(
                f"LOG:select_best_trial: Selected via Tier 2 criteria"
                f"(DI in [{di_tier2_min:.2f}, {di_tier2_max:.2f}), highest DI, then highest F1). "
                f"F1: {tier2_candidates[0]['f1']:.4f}, DI: {tier2_candidates[0]['di']:.4f}"
            )

    # Fallback
    if not selected_trial_obj:
        print(
            "LOG:select_best_trial: WARNING - No trials met Tier 1 or Tier 2 criteria."
            "Applying fallback: highest overall F1."
        )
        # valid_trials_info already contains all trials with valid metrics
        if valid_trials_info: # Should be true if the first valid_trials_info check passed
            valid_trials_info.sort(key=lambda x: x["f1"], reverse=True)
            selected_trial_obj = valid_trials_info[0]["trial_obj"]
            print(
                f"LOG:select_best_trial: Selected via General Fallback (highest F1)."
                f"F1: {valid_trials_info[0]['f1']:.4f}, DI: {valid_trials_info[0]['di']:.4f}"
            )

    if not selected_trial_obj:
        # This case should only occur if valid_trials_info was empty initially,
        # which is already handled at the beginning.
        print("LOG:select_best_trial: CRITICAL ERROR - Unable to select any trial.")

    return selected_trial_obj