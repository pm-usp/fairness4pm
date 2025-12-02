# Libraries for data manipulation
import pandas as pd
import numpy as np
import random
from typing import Optional, List, Dict, Any
import sys
import os

#Libraries for machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

# TensorFlow (required for AdversarialDebiasing)
import tensorflow.compat.v1 as tf

# Libraries for fairness
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# Library for hyperparameter optimization
import optuna

# Library for explainability
from lime.lime_tabular import LimeTabularExplainer

# Library for fairness metrics
from FairnessProcessMining import Metrics

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
    
    dataset_train = BinaryLabelDataset(df = df_for_aif360_train,
                                       label_names=[y_train.name],
                                       protected_attribute_names=[protected_attribute],
                                       favorable_label=1,
                                       unfavorable_label=0)

    df_for_aif360_test = X_test.copy()
    df_for_aif360_test[y_test.name] = y_test 
    
    dataset_test = BinaryLabelDataset(df = df_for_aif360_test,
                                      label_names=[y_test.name],
                                      protected_attribute_names=[protected_attribute],
                                      favorable_label=1,
                                      unfavorable_label=0)

    return dataset_train, dataset_test


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




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                          Pre-processing Models                           """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def PreReweighingRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    This function applies the AIF360 Reweighing technique to the training dataset and then 
    optimizes the hyperparameters of a RandomForestClassifier model using Optuna, 
    optimizing F1 and seeking a Disparate Impact close to 1.
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Combined training and testing DataFrame with predictions and source information.
    - best_model (RandomForestClassifier): The best model trained using the best hyperparameters found.
    - best_params (dict): Dictionary containing the best hyperparameters found by the optimization.
    - best_score (float): The best average F1 score found during the optimization.
    - explanations (list): LIME explanations for the first 10 samples of the test set.
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'

    # Create datasets for the AIF360 library
    dataset_train, dataset_test = AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute)

    # Applies the AIF360 Reweighing technique
    RW = Reweighing(unprivileged_groups=[{protected_attribute: 1}],
                    privileged_groups=[{protected_attribute: 0}])
    dataset_transf_train = RW.fit_transform(dataset_train) 
    
    # Ensure that 'attributes' reflects the features actually used after AIF360, if any change
    # If AIF360Datasets and Reweighing do not change the feature_names of X_train, 'attributes' remains valid.
    # Otherwise, use: current_model_features = dataset_transf_train.feature_names
    current_model_features = list(dataset_transf_train.feature_names)
    if attributes != current_model_features:
        print(f"Warning: Original feature list ('attributes') changed after AIF360 transformations. Using: {current_model_features}")
        attributes = current_model_features # Update 'attributes'

    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    
    # Ensure index consistency
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name=label_col_name, index=X_train_transf.index)
    weights = pd.Series(dataset_transf_train.instance_weights, index=X_train_transf.index)

    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
    
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED,
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
        
        for train_index, val_index in skf.split(X_train_transf, y_train_transf):
            X_fold_train, X_fold_val = X_train_transf.iloc[train_index], X_train_transf.iloc[val_index]
            y_fold_train, y_fold_val = y_train_transf.iloc[train_index], y_train_transf.iloc[val_index]
            weights_fold_train = weights.iloc[train_index] # Usar .iloc se weights for Series com índice não padrão
            
            clf_rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
            
            y_pred_val = clf_rf.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            df_val = X_fold_val.copy() # Mais seguro que recriar com pd.DataFrame e columns=attributes
            df_val[label_col_name] = y_fold_val.values # Usar label_col_name
            df_val['prediction'] = y_pred_val
            
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
                
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- OPTUNA STUDY VIEW ---
    print("\n--- Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0: # Check if there are trials to plot
        try:
            # Pareto Frontier Chart
            # Objective names are inferred from the order returned in 'objective'
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio", "Fairness Score Médio"])
            fig_pareto.show() 
            # To save: fig_pareto.write_html("PreReweighing_pareto_front.html")

            # Optimization History
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # To save: fig_history.write_html("PreReweighing_optimization_history.html")

            # Slice Plot to see the impact of each hyperparameter
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # To save: fig_slice.write_html("PreReweighing_slice_plot.html")

            # Importance of Parameters for F1-score
            fig_importance_f1 = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[0] if t.values is not None else float('nan'), # Primeiro valor retornado pela 'objective' (avg_f1)
                target_name="Importância para F1-score"
            )
            fig_importance_f1.show()
            # To save: fig_importance_f1.write_html("PreReweighing_param_importances_f1.html")

            # Importance of Parameters for the Fairness Score
            fig_importance_fs = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[1] if t.values is not None else float('nan'), # Segundo valor retornado pela 'objective' (fairness_score)
                target_name="Importância para Fairness Score"
            )
            fig_importance_fs.show()
            # To save: fig_importance_fs.write_html("PreReweighing_param_importances_fs.html")

        except Exception as e:
            print(f"Error generating Optuna views: {e}")
            print("Check that Plotly is installed and that there are enough valid trials in the study.")
    else:
        print("No trial in the study to generate views.")

    # --- CUSTOM SELECTION OF THE BEST TRIAL ---
    print("\n--- Starting Custom Best Trial Selection ---")
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best Hyperparameters: {best_params}")

        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1)
        best_model.fit(X_train_transf, y_train_transf, sample_weight=weights)
        
        y_pred_train = best_model.predict(X_train_transf)
        # Original X_test, not transformed by Reweighing
        y_pred_test = best_model.predict(X_test) 
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)
        
        # Generate LIME explanations for first 10 test instances
        explanations = []
        lime_explainer_for_transformed = LimeTabularExplainer(
            training_data=X_train_transf.values, # Explain model that operates on transformed data
            feature_names=attributes, # Column names of X_train_transf
            class_names=[label_col_name if label_col_name else 'Negative', 'Positive'], 
            discretize_continuous=True,
            random_state=SEED
        )

        for i in range(min(10, X_test.shape[0])):
            instance_to_explain = X_test.iloc[i].values 
            
            exp = lime_explainer_for_transformed.explain_instance(
                data_row=instance_to_explain, # Deve ter o mesmo número de features que X_train_transf
                predict_fn=lambda x_lime: best_model.predict_proba(pd.DataFrame(x_lime, columns=attributes)),
                num_features=len(attributes) 
            )
            explanations.append(exp)
        
        return df_final, best_model, best_params, final_f1, explanations

    else:
        print(
            "\nERROR in main script: Unable to select a trial with the defined criteria. "
            "Optimized training cannot proceed."
        )
        return None, None, None, None, None
   

def PreDisparateImpactRemoverRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    This funcion applies the AIF360 Disparate Impact Remover (DIR) technique to the training dataset and then 
    optimizes the hyperparameters of a RandomForestClassifier model using Optuna, 
    optimizing F1 and seeking a Disparate Impact close to 1.
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Final DataFrame with the combined training and testing predictions.
    - best_model (RandomForestClassifier): Best trained model after optimization.
    - best_params (dict): Optimized hyperparameters.
    - best_score (float): Best score resulting from optimization.
    - explanations (list): LIME explanations for the predictions.
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target' 

    # Create datasets for the AIF360 library
    dataset_train, dataset_test = AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute)

    # Applies the AIF360 Disparate Impact Remover (DIR)
    DIR = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=protected_attribute)  # Complete repair
    dataset_transf_train = DIR.fit_transform(dataset_train) 
    
    # Ensure that 'attributes' reflects the features actually used after AIF360, if any change
    # If AIF360Datasets and Reweighing do not change the feature_names of X_train, 'attributes' remains valid.
    # Otherwise, use: current_model_features = dataset_transf_train.feature_names
    current_model_features = list(dataset_transf_train.feature_names)
    if attributes != current_model_features:
        print(f"Warning: Original feature list ('attributes') changed after AIF360 transformations. Using: {current_model_features}")
        attributes = current_model_features # Update 'attributes'

    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    # Ensure index consistency
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name=label_col_name, index=X_train_transf.index)

    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
    
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED,
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
        
        for train_index, val_index in skf.split(X_train_transf, y_train_transf):
            X_fold_train, X_fold_val = X_train_transf.iloc[train_index], X_train_transf.iloc[val_index]
            y_fold_train, y_fold_val = y_train_transf.iloc[train_index], y_train_transf.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            
            y_pred_val = clf_rf.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            df_val = X_fold_val.copy() # Mais seguro que recriar com pd.DataFrame e columns=attributes
            df_val[label_col_name] = y_fold_val.values # Usar label_col_name
            df_val['prediction'] = y_pred_val
            
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
                
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- OPTUNA STUDY VIEW ---
    print("\n---  Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0: # Check if there are trials to plot
        try:
            # Pareto Frontier Chart
            # Objective names are inferred from the order returned in 'objective'
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio", "Fairness Score Médio"])
            fig_pareto.show() 
            # To save: fig_pareto.write_html("PreReweighing_pareto_front.html")

            # Optimization History
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # To save: fig_history.write_html("PreReweighing_optimization_history.html")

            # Slice Plot to see the impact of each hyperparameter
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # To save: fig_slice.write_html("PreReweighing_slice_plot.html")

            # Importance of Parameters for F1-score
            fig_importance_f1 = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[0] if t.values is not None else float('nan'), # Primeiro valor retornado pela 'objective' (avg_f1)
                target_name="Importância para F1-score"
            )
            fig_importance_f1.show()
            # To save: fig_importance_f1.write_html("PreReweighing_param_importances_f1.html")

            # Importance of Parameters for the Fairness Score
            fig_importance_fs = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[1] if t.values is not None else float('nan'), # Segundo valor retornado pela 'objective' (fairness_score)
                target_name="Importância para Fairness Score"
            )
            fig_importance_fs.show()
            # To save: fig_importance_fs.write_html("PreReweighing_param_importances_fs.html")

        except Exception as e:
            print(f"Error generating Optuna views: {e}")
            print("Check that Plotly is installed and that there are enough valid trials in the study.")
    else:
        print("No trial in the study to generate views.")

    # --- CUSTOM SELECTION OF THE BEST TRIAL ---
    print("\n--- Starting Custom Best Trial Selection ---")
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params
        final_f1 = best_trial_found.user_attrs.get("avg_f1") 
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best Hyperparameters: {best_params}")

        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1) 
        best_model.fit(X_train_transf, y_train_transf)
        
        y_pred_train = best_model.predict(X_train_transf)
        # Original X_test, not transformed by DIR
        y_pred_test = best_model.predict(X_test) 
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)
        
        # Generate LIME explanations for first 10 test instances    
        explanations = []
        lime_explainer_for_transformed = LimeTabularExplainer(
            training_data=X_train_transf.values, # Explain model that operates on transformed data
            feature_names=attributes,  # Column names of X_train_transf
            class_names=[label_col_name if label_col_name else 'Negative', 'Positive'],
            discretize_continuous=True,
            random_state=SEED
        )

        for i in range(min(10, X_test.shape[0])):
            instance_to_explain = X_test.iloc[i].values 
            
            exp = lime_explainer_for_transformed.explain_instance(
                data_row=instance_to_explain, 
                predict_fn=lambda x_lime: best_model.predict_proba(pd.DataFrame(x_lime, columns=attributes)),
                num_features=len(attributes) 
            )
            explanations.append(exp)
        
        return df_final, best_model, best_params, final_f1, explanations

    else:
        print(
            "\nERROR in main script: Unable to select a trial with the defined criteria. "
            "Optimized training cannot proceed."
        )
        return None, None, None, None, None
    


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                       In-processing Models                               """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Required for compatibility with TensorFlow 1.x and AIF360
tf.compat.v1.disable_v2_behavior()

class SuppressPrints: 
    """
    Context Manager to Suppress Standard Output (Prints)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class AdversarialDebiasingWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for AdversarialDebiasing with internal MinMaxScaler scaling.
    """
    def __init__(self, protected_attribute_name, privileged_groups, unprivileged_groups, 
                 scope_name='debiased_classifier', debias=True, num_epochs=100, 
                 classifier_num_hidden_units=100, adversary_loss_weight=0.1, 
                 seed=None, 
                 class_attr_name='Target'): 
        
        self.protected_attribute_name = protected_attribute_name
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.scope_name = scope_name
        self.debias = debias
        self.num_epochs = num_epochs
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.adversary_loss_weight = adversary_loss_weight
        self.seed = seed
        self.class_attr_name = class_attr_name 
        
        self.sess = None
        self.model_ = None 
        self.columns_ = None
        self.scaler_ = None 
        self.classes_ = None # For compatibility with ClassifierMixin

    def _set_seeds_if_needed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            tf.compat.v1.set_random_seed(self.seed) # AdversarialDebiasing uses TensorFlow

    def fit(self, X, y):
        self._set_seeds_if_needed()
        self.columns_ = X.columns.tolist()
        self.classes_ = np.unique(y) # Required for ClassifierMixin

        # 1. Feature Scaling
        self.scaler_ = MinMaxScaler()
        X_scaled_np = self.scaler_.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled_np, columns=self.columns_, index=X.index)

        # Ensure y has the expected label column name
        y_named = y.copy()
        if not isinstance(y_named, pd.Series): # If y is numpy array
            y_named = pd.Series(y_named, index=X.index)
        if y_named.name is None:
            y_named.name = self.class_attr_name
        
        aif360_dataset = prepare_aif360_data(X_scaled_df, y_named, self.protected_attribute_name)
        
        tf.compat.v1.reset_default_graph() # Reset graph for each fit (important with TF1.x)
        self._set_seeds_if_needed()      # Reset seeds after graph reset
        self.sess = tf.compat.v1.Session()
        
        self.model_ = AdversarialDebiasing( # Assign to self.model_
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            scope_name=self.scope_name + str(np.random.randint(100000)), # Unique scope name to avoid conflicts
            debias=self.debias,
            num_epochs=self.num_epochs,
            classifier_num_hidden_units=self.classifier_num_hidden_units,
            adversary_loss_weight=self.adversary_loss_weight,
            sess=self.sess,
            seed=self.seed 
        )
        self.model_.fit(aif360_dataset) # Model_ is from AIF360
        return self

    def _prepare_data_for_predict(self, X_input):
        if not isinstance(X_input, pd.DataFrame):
            if isinstance(X_input, np.ndarray):
                X_df = pd.DataFrame(X_input, columns=self.columns_)
            else:
                raise TypeError("Input for predict/predict_proba must be DataFrame or NumPy array.")
        else:
            X_df = X_input.copy() # Ensures that it does not modify the original

        # Check if columns are in the correct order (if X_df already has columns)
        if list(X_df.columns) != self.columns_:
            print("Warning: Columns in the input DataFrame do not match or are not in the order of the training columns. Reordering.")
            X_df = X_df[self.columns_]

        if self.scaler_ is None:
            raise ValueError("Untrained scaler. Call fit first.")
        X_scaled_np = self.scaler_.transform(X_df)
        X_scaled_df = pd.DataFrame(X_scaled_np, columns=self.columns_, index=X_df.index)
        
        y_dummy = pd.Series(np.zeros(X_df.shape[0]), name=self.class_attr_name, index=X_df.index)
        return prepare_aif360_data(X_scaled_df, y_dummy, self.protected_attribute_name)

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Untrained model. Call fit first.")
        aif360_dataset_pred = self._prepare_data_for_predict(X)
        return self.model_.predict(aif360_dataset_pred).labels.ravel()

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Untrained model. Call fit first.")
        aif360_dataset_pred = self._prepare_data_for_predict(X)
        
        pred_results = self.model_.predict(aif360_dataset_pred)
        scores = pred_results.scores.ravel()
        proba_favorable = 1 / (1 + np.exp(-scores))
        proba = np.vstack([1 - proba_favorable, proba_favorable]).T
        
        if not np.allclose(proba.sum(axis=1), 1.0):
            proba_sum = proba.sum(axis=1, keepdims=True)
            proba_sum[proba_sum == 0] = 1 
            proba = proba / proba_sum
        return proba

    def close_session(self):
        if self.sess is not None:
            self.sess.close()
            #TensorFlow session closed for AdversarialDebiasingWrapper.
            self.sess = None # Avoid closing multiple times
    
    def __del__(self): # Ensures that the session is closed when the object is destroyed
        self.close_session()


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



def InAdversarialDebiasingOptuna(X_train, y_train, X_test, y_test, 
                                 df_train, df_test, 
                                 protected_attribute, num_trials=50):
    """
    This function implements the AIF360 Adversarial Debiasing model to mitigate bias during model training. 
    Hyperparameters model are optimized using Optuna, balancing F1-score and fairness (Disparate Impact).
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Combined training and testing DataFrame with predictions and source information.
    - best_model (RandomForestClassifier): The best model trained using the best hyperparameters found.
    - best_params (dict): Dictionary containing the best hyperparameters found by the optimization.
    - best_score (float): The best average F1 score found during the optimization.
    - explanations (list): LIME explanations for the first 10 samples of the test set.
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'

    # LIME Explicador é inicializado com dados ORIGINAIS (não escalados) 
    # A função predizer_proba do wrapper lidará com o escalonamento internamente
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=attributes, 
        class_names=['Negative', 'Positive'], 
        discretize_continuous=True,
        random_state=SEED
    )

    def objective(trial):
        # Hyperparameter search space
        num_epochs = trial.suggest_categorical('num_epochs', [50, 100, 150]) 
        units = trial.suggest_categorical('classifier_num_hidden_units', [50, 100]) 
        weight = trial.suggest_float('adversary_loss_weight', 0.05, 2.0, step=0.05) 
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) 
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train_orig, X_fold_val_orig = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            # Scaling within the fold
            scaler_fold = MinMaxScaler()
            X_fold_train_scaled_np = scaler_fold.fit_transform(X_fold_train_orig)
            X_fold_val_scaled_np = scaler_fold.transform(X_fold_val_orig)

            X_fold_train = pd.DataFrame(X_fold_train_scaled_np, columns=X_fold_train_orig.columns, index=X_fold_train_orig.index)
            X_fold_val = pd.DataFrame(X_fold_val_scaled_np, columns=X_fold_val_orig.columns, index=X_fold_val_orig.index)
            
            y_fold_train_named = y_fold_train.copy()
            if y_fold_train_named.name is None: y_fold_train_named.name = label_col_name
            if not y_fold_train_named.index.equals(X_fold_train.index):
                 y_fold_train_named = pd.Series(y_fold_train_named.values, index=X_fold_train.index, name=y_fold_train_named.name)

            y_fold_val_named = y_fold_val.copy()
            if y_fold_val_named.name is None: y_fold_val_named.name = label_col_name
            if not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val_named.values, index=X_fold_val.index, name=y_fold_val_named.name)

            # The AdversarialDebiasingWrapper handles scaling internally in its fit
            model = AdversarialDebiasingWrapper(
                protected_attribute_name=protected_attribute,
                privileged_groups=[{protected_attribute: 0}],
                unprivileged_groups=[{protected_attribute: 1}],
                num_epochs=num_epochs,
                classifier_num_hidden_units=units,
                adversary_loss_weight=weight,
                seed=SEED,
                class_attr_name=label_col_name
            )
            
            # Wrapper.fit expects X, y Pandas. It will do the internal scaling and prepare_aif360_data.
            with SuppressPrints():
                model.fit(X_fold_train, y_fold_train_named) # Pass original fold data, scale wrapper
                
            # Wrapper scales X_fold_val before predict
            y_pred_val = model.predict(X_fold_val) 
            model.close_session()

            f1 = f1_score(y_fold_val_named, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            # df_val_metric uses original (unscaled) X_fold_val to calculate DI
            # because DI must be calculated on the original features if they are part of the decision
            df_val_metric = X_fold_val.copy() 
            df_val_metric[label_col_name] = y_fold_val_named.values
            df_val_metric['prediction'] = y_pred_val
            
            di = Metrics.ModelDisparateImpact(df_val_metric, protected_attribute, 
                                             privileged_group=0, unprivileged_group=1,
                                             prediction='prediction')
            di_scores.append(di)
            
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10) 
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED), 
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- OPTUNA STUDY VIEW ---
    print("\n--- Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0:
        try:
            # Pareto Frontier Chart
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio CV", "Fairness Score Médio CV"])
            fig_pareto.show()
            # To save: fig_pareto.write_html("AdversarialDebiasing_pareto_front.html")

            # Optimization History
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # To save: fig_history.write_html("AdversarialDebiasing_optimization_history.html")

            # Slice Plot to see the impact of each hyperparameter
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # To save: fig_slice.write_html("AdversarialDebiasing_slice_plot.html")

            # Importance of Parameters for F1-score
            fig_importance_f1 = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[0] if t.values is not None else float('nan'), # Primeiro valor retornado pela 'objective' (avg_f1)
                target_name="Importância para F1-score"
            )
            fig_importance_f1.show()
            # To save: fig_importance_f1.write_html("AdversarialDebiasing_param_importances_f1.html")

            # Importance of Parameters for the Fairness Score
            fig_importance_fs = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[1] if t.values is not None else float('nan'), # Segundo valor retornado pela 'objective' (fairness_score)
                target_name="Importância para Fairness Score"
            )
            fig_importance_fs.show()
            # To save: fig_importance_fs.write_html("AdversarialDebiasing_param_importances_fs.html")

        except Exception as e:
            print(f"Error generating Optuna views: {e}")
            print("Check that Plotly is installed and that there are enough valid trials in the study.")

    else:
        print("No trial in the study to generate views.")

    # --- CUSTOM SELECTION OF THE BEST TRIAL ---
    print("\n--- Starting Custom Best Trial Selection ---")    
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        final_best_params_to_return = best_trial_found.params.copy()
        params_for_model = best_trial_found.params 

        best_score_f1_cv = best_trial_found.user_attrs.get("avg_f1")
        final_di_cv = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if best_score_f1_cv is not None: 
            print(f"F1-score (CV of the selected trial): {best_score_f1_cv:.4f}")
        if final_di_cv is not None: 
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di_cv:.4f}")
        print(f"Best Hyperparameters: {final_best_params_to_return}")
        
        best_model = AdversarialDebiasingWrapper(
            protected_attribute_name=protected_attribute,
            privileged_groups=[{protected_attribute: 0}],
            unprivileged_groups=[{protected_attribute: 1}],
            **params_for_model, 
            seed=SEED,
            class_attr_name=label_col_name
        )
        
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        
        # The best_model.fit wrapper handles the scaling of X_train internally
        with SuppressPrints():
            best_model.fit(X_train, y_train_named) 
            
        # The best_model.predict wrapper handles the scaling of X_train/X_test internally
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test) 
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        # Generate LIME explanations for first 10 test instances
        explanations = []
        for i in range(min(10, len(X_test))):
            instance_to_explain = X_test.iloc[i].values # LIME recebe array NumPy
            
            exp = explainer.explain_instance(
                data_row=instance_to_explain, 
                predict_fn=best_model.predict_proba, # Wrapper lida com scaling e formatação
                num_features=len(attributes) 
            )
            explanations.append(exp)
        
        best_model.close_session() 
        
        return df_final, best_model, final_best_params_to_return, best_score_f1_cv, explanations
    else:
        print("\nERRO: Não foi possível selecionar um trial com os critérios definidos...")
        return None, None, None, None, None



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Pos-processing Models                        """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def PostEOddsPostprocessingRandomFlorestOptuna(X_train, y_train, X_test, y_test, 
                                               df_train, df_test, 
                                               protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    Equalized Odds Postprocessing (EqOddsPostprocessing) to adjust predictions.
    Hyperparameters model are optimized using Optuna, balancing F1-score and fairness (Disparate Impact).
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Combined training and testing DataFrame with predictions and source information.
    - best_model (RandomForestClassifier): The best model trained using the best hyperparameters found.
    - best_params (dict): Dictionary containing the best hyperparameters found by the optimization.
    - best_score (float): The best average F1 score found during the optimization.
    - explanations (list): LIME explanations for the first 10 samples of the test set.
    """    
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'
        
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
            
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            y_val_pred = clf_rf.predict(X_fold_val)
            
            # Ensure consistent names and indexes for Series of labels
            y_fold_val_named = y_fold_val.copy()
            if not isinstance(y_fold_val_named, pd.Series) or y_fold_val_named.name is None:
                 y_fold_val_named = pd.Series(y_fold_val, name=label_col_name, index=X_fold_val.index)
            elif not y_fold_val_named.index.equals(X_fold_val.index): # Garante alinhamento se já for Series
                 y_fold_val_named = pd.Series(y_fold_val.values, name=y_fold_val_named.name, index=X_fold_val.index)
            
            y_val_pred_series = pd.Series(y_val_pred, index=X_fold_val.index, name=label_col_name)
            
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val_named, protected_attribute)
            bd_pred = prepare_aif360_data(X_fold_val, y_val_pred_series, protected_attribute)

            is_priv = (bd_true.protected_attributes.ravel() == 0)
            is_unpriv = (bd_true.protected_attributes.ravel() == 1)
            is_neg_label = (bd_true.labels.ravel() == 0)
            
            num_priv_neg_actual = np.sum(is_priv & is_neg_label)
            num_unpriv_neg_actual = np.sum(is_unpriv & is_neg_label)
            
            MIN_NEGATIVES_PER_GROUP = 1
            f1_fold = 0.0
            di_fold = 0.0 

            if num_priv_neg_actual < MIN_NEGATIVES_PER_GROUP or \
               num_unpriv_neg_actual < MIN_NEGATIVES_PER_GROUP:
                #Fold: Insufficient negative instances.
                pass 
            else:
                try:
                    eq = EqOddsPostprocessing(
                        unprivileged_groups=[{protected_attribute: 1}],
                        privileged_groups=[{protected_attribute: 0}],
                        seed=SEED
                    )
                    eq.fit(bd_true, bd_pred)
                    bd_pred_post = eq.predict(bd_pred)
                    y_post_val = bd_pred_post.labels.ravel()

                    f1_fold = f1_score(y_fold_val_named, y_post_val, average='weighted')
                    
                    df_val_metric = X_fold_val.copy()
                    df_val_metric[label_col_name] = y_fold_val_named.values
                    df_val_metric['prediction'] = y_post_val
                    
                    di_fold = Metrics.ModelDisparateImpact(df_val_metric, protected_attribute, 
                                                           privileged_group=0, unprivileged_group=1,
                                                           prediction='prediction')
                except ValueError as e:
                    if "c must not contain values inf, nan, or None" in str(e) or \
                       "Unable to solve optimization problem" in str(e):
                        # Fold: Error in linprog
                        pass 
                    else:
                        # Fold: Unexpected ValueError
                        pass 
                except ZeroDivisionError as e:
                    # Fold: ZeroDivisionError 
                    pass 
            
            f1_scores.append(f1_fold)
            di_scores.append(di_fold)
            
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)
    
    # --- OPTUNA STUDY VIEW ---
    print("\n--- Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0:
        try:
            # Pareto Frontier Chart
            # Objective names are inferred from the order returned in 'objective'
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio CV", "Fairness Score Médio CV"])
            fig_pareto.show()
            # To save: fig_pareto.write_html("PostEOdds_pareto_front.html")

            # Optimization History
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # To save: fig_history.write_html("PostEOdds_optimization_history.html")

            # Slice Plot to see the impact of each hyperparameter
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # To save: fig_slice.write_html("PostEOdds_slice_plot.html")

            # Importance of Parameters for F1-score
            valid_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
            if valid_trials_for_importance:
                fig_importance_f1 = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[0] if t.values is not None and len(t.values) > 0 else float('nan'), 
                    target_name="Importance for F1-score CV"
                )
                fig_importance_f1.show()

                fig_importance_fs = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[1] if t.values is not None and len(t.values) > 1 else float('nan'), 
                    target_name="Importance for CV Fairness Score"
                )
                fig_importance_fs.show()
            else:
                print("No complete trial with valid values ​​to generate importance graphs.")

        except Exception as e:
            print(f"Error generating Optuna views: {e}")
            print("Check that Plotly is installed and that there are enough valid trials in the study.")
    else:
        print("No trial in the study to generate views.")

    # --- CUSTOM SELECTION OF THE BEST TRIAL ---
    print("\n--- Starting Custom Best Trial Selection ---")    
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params # Contains only RF params for this model
        best_score = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if best_score is not None: 
            print(f"F1-score (CV of the selected trial): {best_score:.4f}")
        if final_di is not None: 
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best Hyperparameters (RandomForest): {best_params}")
        
        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        y_train_raw = best_model.predict(X_train)
        y_test_raw  = best_model.predict(X_test)
        
        # Ensure names and indices for Series of labels and raw predictions
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        
        y_test_named = y_test.copy() # Usado para bd_test_pred
        if y_test_named.name is None: y_test_named.name = label_col_name

        y_train_raw_series = pd.Series(y_train_raw, index=y_train_named.index, name=label_col_name)
        y_test_raw_series = pd.Series(y_test_raw, index=y_test_named.index, name=label_col_name)

        bd_train_true = prepare_aif360_data(X_train, y_train_named, protected_attribute)
        bd_train_pred = prepare_aif360_data(X_train, y_train_raw_series, protected_attribute)
        bd_test_pred  = prepare_aif360_data(X_test,  y_test_raw_series, protected_attribute)
        
        eq_odds = EqOddsPostprocessing(
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}],
            seed=SEED
        )
        
        y_train_pred_final = y_train_raw 
        y_test_pred_final = y_test_raw   
        lime_predict_fn = lambda X_lime_input_df: best_model.predict_proba(X_lime_input_df)

        try:
            # Fine-tuning the final EqOddsPostprocessing on the full training set
            eq_odds.fit(bd_train_true, bd_train_pred)
            # Applying final EqOddsPostprocessing to training and testing data
            y_train_pred_post = eq_odds.predict(bd_train_pred).labels.ravel()
            y_test_pred_post  = eq_odds.predict(bd_test_pred).labels.ravel()
            
            y_train_pred_final = y_train_pred_post 
            y_test_pred_final = y_test_pred_post   

            def post_processed_predict_proba(X_lime_input_np):
                X_lime_df = pd.DataFrame(X_lime_input_np, columns=attributes)
                probs_raw = best_model.predict_proba(X_lime_df)
                labels_raw = np.argmax(probs_raw, axis=1)
                labels_raw_series = pd.Series(labels_raw, index=X_lime_df.index, name=label_col_name)
                
                bd_lime = prepare_aif360_data(X_lime_df, labels_raw_series, protected_attribute)
                bd_lime_post = eq_odds.predict(bd_lime) 
                return np.vstack([1 - bd_lime_post.scores.ravel(), bd_lime_post.scores.ravel()]).T
            
            lime_predict_fn = post_processed_predict_proba

        except ValueError as e:
            if "c must not contain values inf, nan, or None" in str(e) or \
               "Unable to solve optimization problem" in str(e):
                print(f"WARNING: Solver error when fitting EqOdds on the full dataset ({e}). Using raw predictions.")
            else:
                print(f"ERROR: Unexpected ValueError during final EqOdds({e}). Using raw predictions.")
        except ZeroDivisionError as e:
            print(f"WARNING: ZeroDivisionError during final EqOdds({e}). Using raw predictions.")
        
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_train_pred_final
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_test_pred_final
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)


        # Generate LIME explanations for first 10 test instances
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(), 
            class_names=['Negative', 'Positive'], 
            discretize_continuous=True,
            random_state=SEED
        )
        
        explanations = [
            explainer.explain_instance(
                X_test.iloc[i].values,
                lime_predict_fn,
                num_features=len(attributes) 
            ) for i in range(min(10, len(X_test)))
        ]
        
        return df_final, best_model, best_params, best_score, explanations
    else:
        print("\nERROR: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None
    


def PostCalibratedEOddsRandomFlorestOptuna(X_train, y_train, X_test, y_test, 
                                           df_train, df_test, 
                                           protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    Calibrated Equalized Odds Postprocessing (CalibratedEqOddsPostprocessing) to adjust predictions.
    Hyperparameters model are optimized using Optuna, balancing F1-score and fairness (Disparate Impact).
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Combined training and testing DataFrame with predictions and source information.
    - best_model (RandomForestClassifier): The best model trained using the best hyperparameters found.
    - best_params (dict): Dictionary containing the best hyperparameters found by the optimization.
    - best_score (float): The best average F1 score found during the optimization.
    - explanations (list): LIME explanations for the first 10 samples of the test set.
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'
    
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
            
        cost_constraint = trial.suggest_categorical("cost_constraint", ["fnr", "fpr", "weighted"])

        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            y_val_pred = clf_rf.predict(X_fold_val)
            
            # Ensure consistent names and indexes for Series of labels
            y_fold_val_named = y_fold_val.copy()
            if not isinstance(y_fold_val_named, pd.Series) or y_fold_val_named.name is None:
                 y_fold_val_named = pd.Series(y_fold_val, name=label_col_name, index=X_fold_val.index)
            elif not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val.values, name=y_fold_val_named.name, index=X_fold_val.index)
            
            y_val_pred_series = pd.Series(y_val_pred, index=X_fold_val.index, name=label_col_name)
            
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val_named, protected_attribute)
            bd_pred = prepare_aif360_data(X_fold_val, y_val_pred_series, protected_attribute)

            is_priv = (bd_true.protected_attributes.ravel() == 0)
            is_unpriv = (bd_true.protected_attributes.ravel() == 1)
            is_neg_label = (bd_true.labels.ravel() == 0)
            
            num_priv_neg_actual = np.sum(is_priv & is_neg_label)
            num_unpriv_neg_actual = np.sum(is_unpriv & is_neg_label)
            
            MIN_NEGATIVES_PER_GROUP = 1
            f1_fold = 0.0
            di_fold = 0.0 

            if num_priv_neg_actual < MIN_NEGATIVES_PER_GROUP or \
               num_unpriv_neg_actual < MIN_NEGATIVES_PER_GROUP:
                #Fold: Insufficient negative instances.
                pass 
            else:
                try:
                    post = CalibratedEqOddsPostprocessing(
                        unprivileged_groups=[{protected_attribute: 1}],
                        privileged_groups=[{protected_attribute: 0}],
                        cost_constraint=cost_constraint,
                        seed=SEED
                    )
                    post.fit(bd_true, bd_pred)
                    bd_pred_post = post.predict(bd_pred)
                    y_post_val = bd_pred_post.labels.ravel()

                    f1_fold = f1_score(y_fold_val_named, y_post_val, average='weighted')
                    
                    df_val_metric = X_fold_val.copy()
                    df_val_metric[label_col_name] = y_fold_val_named.values
                    df_val_metric['prediction'] = y_post_val
                    
                    di_fold = Metrics.ModelDisparateImpact(df_val_metric, protected_attribute, 
                                                           privileged_group=0, unprivileged_group=1,
                                                           prediction='prediction')
                except ValueError as e:
                    if "c must not contain values inf, nan, or None" in str(e) or \
                       "Unable to solve optimization problem" in str(e) or \
                       "constraint violation" in str(e).lower():
                        # Fold: Solver error
                        pass 
                    else:
                        # Fold: Unexpected ValueError
                        pass 
                except ZeroDivisionError as e:
                    # Fold: ZeroDivisionError 
                    pass 
            
            f1_scores.append(f1_fold)
            di_scores.append(di_fold)
            
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)
    
    # --- OPTUNA STUDY VIEW ---
    print("\n--- Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0:
        try:
            # Pareto Frontier Chart
            # Objective names are inferred from the order returned in 'objective'
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio CV", "Fairness Score Médio CV"])
            fig_pareto.show()
            # To save: fig_pareto.write_html("PostCalibEOdds_pareto_front.html") # Nome de arquivo específico

            # Optimization History
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # To save: fig_history.write_html("PostCalibEOdds_optimization_history.html")

            # Slice Plot to see the impact of each hyperparameter
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # To save: fig_slice.write_html("PostCalibEOdds_slice_plot.html")

            # Importance of Parameters for F1-score
            valid_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
            if valid_trials_for_importance:
                fig_importance_f1 = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[0] if t.values is not None and len(t.values) > 0 else float('nan'), 
                    target_name="Importance for F1-score CV"
                )
                fig_importance_f1.show()

                fig_importance_fs = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[1] if t.values is not None and len(t.values) > 1 else float('nan'), 
                    target_name="Importance for CV Fairness Score"
                )
                fig_importance_fs.show()
            else:
                print("No complete trial with valid values ​​to generate importance graphs.")

        except Exception as e:
            print(f"Error generating Optuna views: {e}")
            print("Check that Plotly is installed and that there are enough valid trials in the study.")
    else:
        print("No trial in the study to generate views.")

    
    # --- CUSTOM SELECTION OF THE BEST TRIAL ---
    print("\n--- Starting Custom Best Trial Selection ---")    
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        final_best_params_to_return = best_trial_found.params.copy()
        
        params_for_model_instantiation = best_trial_found.params 
        cost_constraint_best = params_for_model_instantiation.pop('cost_constraint')
        rf_params_best = params_for_model_instantiation # Contains only RF params for this model
        
        best_score_cv = best_trial_found.user_attrs.get("avg_f1")
        final_di_cv = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal (Baseada na CV) ---")
        if best_score_cv is not None: 
            print(f"F1-score (CV of the selected trial): {best_score_cv:.4f}")
        if final_di_cv is not None: 
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di_cv:.4f}")
        print(f"Best  cost_constraint (CV): {cost_constraint_best}") 
        print(f"Best Hyperparameters (RandomForest): {rf_params_best}")

        best_model = RandomForestClassifier(**rf_params_best, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        y_train_raw = best_model.predict(X_train)
        y_test_raw  = best_model.predict(X_test)
        
        # Ensure names and indices for Series of labels and raw predictions
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        
        y_test_named = y_test.copy()
        if y_test_named.name is None: y_test_named.name = label_col_name

        y_train_raw_series = pd.Series(y_train_raw, index=y_train_named.index, name=label_col_name)
        y_test_raw_series = pd.Series(y_test_raw, index=y_test_named.index, name=label_col_name)

        bd_train_true = prepare_aif360_data(X_train, y_train_named, protected_attribute)
        bd_train_pred = prepare_aif360_data(X_train, y_train_raw_series, protected_attribute)
        bd_test_pred  = prepare_aif360_data(X_test,  y_test_raw_series, protected_attribute)
        
        final_post = CalibratedEqOddsPostprocessing(
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}],
            cost_constraint=cost_constraint_best, 
            seed=SEED,
        )
        
        y_train_pred_final = y_train_raw 
        y_test_pred_final = y_test_raw   
        lime_predict_fn = lambda X_lime_input_df: best_model.predict_proba(X_lime_input_df) 

        try:
            # Fine-tuning the final CalibratedEqOddsPostprocessing on the full training set
            final_post.fit(bd_train_true, bd_train_pred)
            # Applying final CalibratedEqOddsPostprocessing to training and testing data
            y_train_pred_post = final_post.predict(bd_train_pred).labels.ravel()
            y_test_pred_post  = final_post.predict(bd_test_pred).labels.ravel()
            
            y_train_pred_final = y_train_pred_post 
            y_test_pred_final = y_test_pred_post   

            def post_processed_predict_proba(X_lime_input_np):
                X_lime_df = pd.DataFrame(X_lime_input_np, columns=attributes)
                probs_raw = best_model.predict_proba(X_lime_df)
                labels_raw = np.argmax(probs_raw, axis=1)
                labels_raw_series = pd.Series(labels_raw, index=X_lime_df.index, name=label_col_name)
                
                bd_lime = prepare_aif360_data(X_lime_df, labels_raw_series, protected_attribute)
                bd_lime_post = final_post.predict(bd_lime) 
                return np.vstack([1 - bd_lime_post.scores.ravel(), bd_lime_post.scores.ravel()]).T
            
            lime_predict_fn = post_processed_predict_proba

        except ValueError as e:
            if "c must not contain values inf, nan, or None" in str(e) or \
               "Unable to solve optimization problem" in str(e) or \
               "constraint violation" in str(e).lower():
                print(f"AVISO: Solver error when fitting CalibratedEqOdds on the full dataset ({e}). Using raw predictions.")
                pass 
            else:
                print(f"ERROR: Unexpected ValueError during final CalibratedEqOdds({e}). Using raw predictions.")
                pass 
        except ZeroDivisionError as e:
            print(f"WARNING: ZeroDivisionError during final CalibratedEqOdds({e}). Using raw predictions.")
            pass 
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_train_pred_final
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_test_pred_final
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        # Generate LIME explanations for first 10 test instances
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(), 
            class_names=['Negative', 'Positive'], 
            discretize_continuous=True,
            random_state=SEED
        )
        
        explanations = [
            explainer.explain_instance(
                X_test.iloc[i].values,
                lime_predict_fn,
                num_features=len(attributes) 
            ) for i in range(min(10, len(X_test)))
        ]
        
        # Retorna o dicionário completo de parâmetros que foi otimizado
        return df_final, best_model, final_best_params_to_return, best_score_cv, explanations 
    else:
        print("\nERROR: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None




