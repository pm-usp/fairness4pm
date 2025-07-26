# Libraries for data manipulation
import pandas as pd
import numpy as np
import random
import sys
import os

#Libraries for machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Libraries for fairness
directory = ''
sys.path.append(os.path.abspath(directory))
from AIF360.preprocessing.Reweighing import Reweighing

from FairnessProcessMining.AIF360.auxiliary import AIF360Datasets
from FairnessProcessMining.AIF360.auxiliary import select_best_hyperparameters_trial

# Library for hyperparameter optimization
import optuna

# Library for explainability
from lime.lime_tabular import LimeTabularExplainer

# Library for fairness metrics
from FairnessProcessMining import Metrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def PreReweighingRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    This funcion applies the AIF360 Reweighing pre-processing technique to the training dataset and then 
    optimizes the hyperparameters of a RandomForestClassifier model using Optuna, 
    balancing F1-score and fairness.
    
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
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_categorical('max_depth', [5, 10, 15, 20, None])
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
            class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        else:
            max_samples_val = None
            class_weight = 'balanced'
               
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap_enabled,
            class_weight=class_weight,
            max_samples=max_samples_val,
            criterion=criterion, 
            n_jobs=1,
            random_state=SEED
        )
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
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
            
            df_val = X_fold_val.copy() 
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
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5) 
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=7), 
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- OPTUNA STUDY VIEW ---
    print("\n--- Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0: # Check if there are trials to plot
        try:
            # Pareto Frontier Chart
            # Objective names are inferred from the order returned in 'objective'
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Average CV F1-score", "Average CV Fairness Score"])
            fig_pareto.show() 
            # To save: fig_pareto.write_html("PreReweighing_pareto_front.html")

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

        # --- Final Training ---      
        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1)
        best_model.fit(X_train_transf, y_train_transf, sample_weight=weights)
        
        # Final predictions
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
        
        # Generate LIME explanations
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
        print("\nERRO: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None
