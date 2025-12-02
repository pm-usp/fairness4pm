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
from AIF360.preprocessing.LFR import LFR

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


def PreLFRRandomForestOptuna(X_train, y_train, X_test, y_test, 
                             df_train, df_test, 
                             protected_attribute, num_trials=50):
    """
    This funcion applies the AIF360 Learning Fair Representations (LFR) pre-processing technique to the training dataset and then 
    optimizes the hyperparameters of a RandomForestClassifier model using Optuna, 
    balancing F1-score and fairness.
    
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
    
    original_attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=original_attributes,
        class_names=['Classe 0', 'Classe 1'],
        discretize_continuous=True,
        random_state=SEED
    )
    
    # AIF360 dataset of the full training, used in the final LFR fit
    dataset_train_orig = prepare_aif360_data(X_train, y_train, protected_attribute)

    def objective(trial):
        # Hyperparameter search space
        # k: dimensionality of the new representation
        lfr_k = trial.suggest_int('lfr_k', 4, 10)
        # Weights for the LFR loss function
        lfr_Ax = trial.suggest_float('lfr_Ax', 0.01, 1.0, log=True) # Weight of Reconstruction - Fidelity
        lfr_Ay = trial.suggest_float('lfr_Ay', 0.01, 1.0, log=True) # Utility Weight - Accuracy
        lfr_Az = trial.suggest_float('lfr_Az', 0.01, 1.0, log=True) # Fairness Weight - Fairness
        
        # RandomForest
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
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train_orig, X_fold_val_orig = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            dataset_train_fold = prepare_aif360_data(X_fold_train_orig, y_fold_train, protected_attribute)
            
            # Train LFR on the training fold
            lfr_fold = LFR(unprivileged_groups=[{protected_attribute: 1}],
                           privileged_groups=[{protected_attribute: 0}],
                           k=lfr_k, Ax=lfr_Ax, Ay=lfr_Ay, Az=lfr_Az,
                           seed=SEED, verbose=0)
            lfr_fold.fit(dataset_train_fold)
            
            # Transforming the training and validation data of the fold
            dataset_train_fold_transf = lfr_fold.transform(dataset_train_fold)
            
            dataset_val_fold_orig = prepare_aif360_data(X_fold_val_orig, y_fold_val, protected_attribute)
            dataset_val_fold_transf = lfr_fold.transform(dataset_val_fold_orig)
            
            X_fold_train_transf = pd.DataFrame(dataset_train_fold_transf.features)
            X_fold_val_transf = pd.DataFrame(dataset_val_fold_transf.features)
            
            # Train RandomForest on the transformed data
            clf_rf.fit(X_fold_train_transf, y_fold_train)
            y_pred_val = clf_rf.predict(X_fold_val_transf)
            
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            # For DI, we use the original features (X_fold_val_orig) for context,
            # but predictions made on the transformed data.
            df_val_metric = X_fold_val_orig.copy()
            df_val_metric[label_col_name] = y_fold_val.values
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
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5) 
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=7), 
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)
    
    # --- OPTUNA STUDY VIEW ---
    print("\n--- Generating Optuna Study Visualizations ---")
    if len(study.trials) > 0:
        try:
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Average CV F1-score", "Average CV Fairness Score"])
            fig_pareto.show()
            # fig_pareto.write_html("PreLFR_pareto_front.html")
            
        except Exception as e:
            print(f"Error generating Optuna views: {e}")
            print("Check that Plotly is installed and that there are enough valid trials in the study.")
    else:
        print("No trial in the study to generate views.")
    

    # --- CUSTOM SELECTION OF THE BEST TRIAL ---
    print("\n--- Starting Custom Best Trial Selection ---") 
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        final_best_params = best_trial_found.params.copy()
        
        # Separate LFR and RandomForest parameters
        lfr_params_best = {k.replace('lfr_', ''): v for k, v in final_best_params.items() if k.startswith('lfr_')}
        rf_params_best = {k: v for k, v in final_best_params.items() if not k.startswith('lfr_')}
        
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
    
        print(f"Best Hyperparameters (LFR + RF): {final_best_params}")
        
        # --- Final Training ---       
        # Train final LFR on the full training set
        lfr_final = LFR(unprivileged_groups=[{protected_attribute: 1}],
                        privileged_groups=[{protected_attribute: 0}],
                        seed=SEED, verbose=0, **lfr_params_best)
        lfr_final.fit(dataset_train_orig)
        
        # Transforming training and testing with the trained LFR
        dataset_train_transf_final = lfr_final.transform(dataset_train_orig)
        X_train_transf_final = pd.DataFrame(dataset_train_transf_final.features)

        dataset_test_orig = prepare_aif360_data(X_test, y_test, protected_attribute)
        dataset_test_transf_final = lfr_final.transform(dataset_test_orig)
        X_test_transf_final = pd.DataFrame(dataset_test_transf_final.features)
        
        # Train final RandomForest on the transformed data
        best_model = RandomForestClassifier(**rf_params_best, random_state=SEED, n_jobs=1)
        best_model.fit(X_train_transf_final, y_train)
        
        # Final predictions
        y_pred_train = best_model.predict(X_train_transf_final)
        y_pred_test = best_model.predict(X_test_transf_final)
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)
        
        # Generate LIME explanations  
        def lfr_rf_predict_proba(X_original_np):
            # Function for LIME that applies the full pipeline to data in its original format
            X_orig_df = pd.DataFrame(X_original_np, columns=original_attributes)
            # Dummy 'y' to create the AIF360 dataset
            y_dummy = pd.Series(np.zeros(X_orig_df.shape[0]), name=label_col_name, index=X_orig_df.index)
            dataset_to_transform = prepare_aif360_data(X_orig_df, y_dummy, protected_attribute)
            
            # Apply the already trained LFR transformation
            dataset_transformed = lfr_final.transform(dataset_to_transform)
            X_transformed = pd.DataFrame(dataset_transformed.features)
            
            # Make the prediction with RandomForest trained on the transformed data
            return best_model.predict_proba(X_transformed)

        explanations = [
            explainer.explain_instance(
                data_row=X_test.iloc[i].values, 
                predict_fn=lfr_rf_predict_proba,
                num_features=len(original_attributes)
            ) for i in range(min(10, len(X_test)))
        ]
        
        return df_final, best_model, final_best_params, final_f1, explanations
    else:
        print("\nERRO: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None    
