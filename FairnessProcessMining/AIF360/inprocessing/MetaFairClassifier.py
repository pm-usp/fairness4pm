# Libraries for data manipulation
import pandas as pd
import numpy as np
import random
import sys
import os

#Libraries for machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Libraries for fairness
directory = ''
sys.path.append(os.path.abspath(directory))
from AIF360.inprocessing.MetaFairClassifier import MetaFairClassifier

from FairnessProcessMining.AIF360.auxiliary import prepare_aif360_data
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



def InMetaFairClassifierOptuna(X_train, y_train, X_test, y_test, 
                               df_train, df_test, 
                               protected_attribute, num_trials=50):
    """
    This function implements the AIF360 MetaFairClassifier model to mitigate bias during model training. 
    Hyperparameters model are optimized using Optuna, balancing F1-score and fairness.
    
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

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=attributes,
        class_names=['Classe 0', 'Classe 1'],
        discretize_continuous=True,
        random_state=SEED
    )

    def objective(trial):
        # Hyperparameter search space
        mfc_tau = trial.suggest_float('mfc_tau', 0.0, 1.0, step=0.05)
        mfc_type = trial.suggest_categorical('mfc_type', ['fdr', 'sr'])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
            
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            f1_fold, di_fold = 0.0, 0.0
            
            try:
                mfc_model = MetaFairClassifier(
                    tau=mfc_tau,
                    sensitive_attr=protected_attribute,
                    type=mfc_type,
                    seed=SEED
                )
                
                dataset_train_fold = prepare_aif360_data(X_fold_train, y_fold_train, protected_attribute)
                mfc_model.fit(dataset_train_fold)
                
                dataset_val_fold = prepare_aif360_data(X_fold_val, y_fold_val, protected_attribute)
                pred_val_dataset = mfc_model.predict(dataset_val_fold)
                y_pred_val = pred_val_dataset.labels.ravel()

                f1_fold = f1_score(y_fold_val, y_pred_val, average='weighted')
                
                df_val_metric = X_fold_val.copy()
                df_val_metric[label_col_name] = y_fold_val.values
                df_val_metric['prediction'] = y_pred_val
                
                di_fold = Metrics.ModelDisparateImpact(df_val_metric, protected_attribute, 
                                                       privileged_group=0, unprivileged_group=1,
                                                       prediction='prediction')
            except Exception as e:
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
        best_params = best_trial_found.params.copy()
                
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best Hyperparameters: {best_params}")
        
        # --- Final Training ---
        # Instantiate the model with the best parameters
        best_model = MetaFairClassifier(
            tau=best_params['mfc_tau'],
            type=best_params['mfc_type'],
            sensitive_attr=protected_attribute,
            seed=SEED
        )
        
        dataset_train_full = prepare_aif360_data(X_train, y_train, protected_attribute)
        best_model.fit(dataset_train_full)
        
        dataset_train_for_pred = prepare_aif360_data(X_train, y_train, protected_attribute)
        dataset_test_for_pred = prepare_aif360_data(X_test, y_test, protected_attribute)
        
        y_pred_train = best_model.predict(dataset_train_for_pred).labels.ravel()
        y_pred_test = best_model.predict(dataset_test_for_pred).labels.ravel()
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        # Generate LIME explanations 
        def mfc_rf_predict_proba(X_original_np):
            X_df = pd.DataFrame(X_original_np, columns=attributes)
            y_dummy = pd.Series(np.zeros(X_df.shape[0]), name=label_col_name, index=X_df.index)
            dataset_to_predict = prepare_aif360_data(X_df, y_dummy, protected_attribute)
            
            try:
                proba = best_model.predict_proba(dataset_to_predict)
                
                # SANITY CHECK: Checks the returned probabilities for NaNs
                if np.isnan(proba).any():
                    np.nan_to_num(proba, copy=False, nan=0.5)
                return proba

            except Exception as e:
                return np.full((X_original_np.shape[0], 2), 0.5)

        explanations = []

        try:
            for i in range(min(10, len(X_test))):
                instance_to_explain = X_test.iloc[i].values
                
                exp = explainer.explain_instance(
                    data_row=instance_to_explain, 
                    predict_fn=mfc_rf_predict_proba,
                    num_features=len(attributes)
                )
                explanations.append(exp)
        except Exception as e:
            print(f"CRITICAL ERROR while generating LIME explanations: {e}")
        
        return df_final, best_model, best_params, final_f1, explanations
    else:
        print("\nERRO: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None


