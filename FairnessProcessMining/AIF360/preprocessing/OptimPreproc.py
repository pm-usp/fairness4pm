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
from AIF360.preprocessing.OptimPreproc import OptimPreproc
from AIF360.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from AIF360.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_custom

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


def discretize_features_custom(df):
    """
    Applies custom discretization, generic binarization, and label encoding 
    to ensure that all features have low cardinality.
    """
    df_processed = df.copy()
    specific_cols = ['case:age', 'case:yearsOfEducation', 'case:CreditScore']

    # --- 1. Specific Discretization Rules ---
    if 'case:age' in df_processed.columns:
        age_bins = [-1, 12, 17, 24, 34, 49, 64, float('inf')]
        age_labels = ['Childhood', 'Adolescence', 'Youth', 'Young Adult', 'Middle-Aged Adult', 'Pre-Retirement', 'Retired']
        df_processed['case:age'] = pd.cut(df_processed['case:age'], bins=age_bins, labels=age_labels, right=True).cat.codes
    
    if 'case:yearsOfEducation' in df_processed.columns:
        edu_bins = [-float('inf'), 0, 8, 12, 16, float('inf')]
        edu_labels = ['No Formal Education', 'Elementary Education', 'Secondary Education', 'Higher Education', 'Postgraduate Education']
        df_processed['case:yearsOfEducation'] = pd.cut(df_processed['case:yearsOfEducation'], bins=edu_bins, labels=edu_labels, right=True).cat.codes
    
    if 'case:CreditScore' in df_processed.columns:
        score_bins = [-1, 20, 40, 60, 80, 100]
        score_labels = ['Low', 'Moderate', 'Medium', 'High', 'Excellent']
        df_processed['case:CreditScore'] = pd.cut(df_processed['case:CreditScore'], bins=score_bins, labels=score_labels, right=True)
        if df_processed['case:CreditScore'].isnull().any():
            df_processed['case:CreditScore'] = df_processed['case:CreditScore'].cat.add_categories('Unknown').fillna('Unknown')
        df_processed['case:CreditScore'] = df_processed['case:CreditScore'].cat.codes
        
    # --- 2. Generic Rules for All Other Columns ---
    for col in df_processed.columns:
        if col in specific_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            unique_values = set(df_processed[col].unique())
            if not unique_values.issubset({0, 1}):
                df_processed[col] = (df_processed[col] > 0).astype(int)
        elif pd.api.types.is_object_dtype(df_processed[col]) or pd.api.types.is_categorical_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].astype('category').cat.codes
            
    return df_processed



def PreOptimPreprocRandomFlorestOptuna(X_train, y_train, X_test, y_test, 
                                       df_train, df_test, 
                                       protected_attribute, num_trials=50):
    """
    This funcion applies the AIF360 Optimized pre-processing technique to the training dataset and then 
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
    
    # --- CUSTOM DISCRETIZATION STAGE
    try:
        X_train_binned = discretize_features_custom(X_train)
        X_test_binned = discretize_features_custom(X_test)
    except Exception as e:
        print(f"CRITICAL ERROR during discretization: {e}")
        return None, None, None, None, None

    # From here on, the function uses X_train_binned and X_test_binned
    attributes = X_train_binned.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'

    # LIME should now be initialized with the discretized data
    explainer = LimeTabularExplainer(
        training_data=X_train_binned.values,
        feature_names=attributes,
        class_names=['Classe 0', 'Classe 1'],
        discretize_continuous=False,
        random_state=SEED
    )
    
    # AIF360 dataset of the full training
    dataset_train_orig = prepare_aif360_data(X_train_binned, y_train, protected_attribute)

    def objective(trial):
        # Hyperparameter search space
        op_gamma = trial.suggest_float('op_gamma', 1e-4, 1.0, log=True)
        
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

        distortion_function = lambda vold, vnew: get_distortion_custom(
            vold, vnew, 
            protected_attribute_name=protected_attribute, 
            label_name=label_col_name
        )
        
        optim_options = {
            "distortion_fun": distortion_function, 
            "epsilon": 0.05, "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, .05, 0], "gamma": op_gamma 
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        f1_scores, di_scores = [], []
    
        for train_index, val_index in skf.split(X_train_binned, y_train):
            X_fold_train, X_fold_val = X_train_binned.iloc[train_index], X_train_binned.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            dataset_train_fold = prepare_aif360_data(X_fold_train, y_fold_train, protected_attribute)
            
            f1_fold, di_fold = 0.0, 0.0

            try:
                OP = OptimPreproc(OptTools, optim_options, seed=SEED, verbose=False)
                OP = OP.fit(dataset_train_fold)
                
                dataset_train_fold_transf = OP.transform(dataset_train_fold, transform_Y=True)
                X_fold_train_transf = pd.DataFrame(dataset_train_fold_transf.features)
                y_fold_train_transf = pd.Series(dataset_train_fold_transf.labels.ravel())

                dataset_val_fold_orig = prepare_aif360_data(X_fold_val, y_fold_val, protected_attribute)
                dataset_val_fold_transf = OP.transform(dataset_val_fold_orig, transform_Y=True)
                X_fold_val_transf = pd.DataFrame(dataset_val_fold_transf.features)
                
                clf_rf.fit(X_fold_train_transf, y_fold_train_transf)
                y_pred_val = clf_rf.predict(X_fold_val_transf)
                
                f1_fold = f1_score(y_fold_val, y_pred_val, average='weighted')
                
                df_val_metric = X_fold_val.copy()
                df_val_metric[label_col_name] = y_fold_val.values
                df_val_metric['prediction'] = y_pred_val
                
                di_fold = Metrics.ModelDisparateImpact(df_val_metric, protected_attribute, 
                                                     privileged_group=0, unprivileged_group=1,
                                                     prediction='prediction')
            except Exception as e:
                # print(f"AVISO - Trial {trial.number}, Fold: Falha no OptimPreproc ({e}). Penalizando.")
                pass # Mantém scores de penalidade f1=0, di=0
            
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
    if len(study.trials) > 0:
        try:
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Average CV F1-score", "Average CV Fairness Score"])
            fig_pareto.show()
            
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
        op_params_best = {k.replace('op_', ''): v for k, v in final_best_params.items() if k.startswith('op_')}
        rf_params_best = {k: v for k, v in final_best_params.items() if not k.startswith('op_')}
        
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
    
        print(f"Best Hyperparameters (Optim + RF): {best_trial_found}")
        
       
        # --- Final Training ---          
        final_distortion_function = lambda vold, vnew: get_distortion_custom(vold, vnew, protected_attribute, label_col_name)
        final_optim_options = {
            "distortion_fun": final_distortion_function, "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99], "dlist": [.1, .05, 0],
            "gamma": op_params_best['gamma']
        }
        OP_final = OptimPreproc(OptTools, final_optim_options, seed=SEED, verbose=False)
        OP_final = OP_final.fit(dataset_train_orig)
        
        # Transform training and test data
        dataset_train_transf_final = OP_final.transform(dataset_train_orig, transform_Y=True)
        X_train_transf_final = pd.DataFrame(dataset_train_transf_final.features, columns=dataset_train_transf_final.feature_names)
        y_train_transf_final = pd.Series(dataset_train_transf_final.labels.ravel())

        dataset_test_orig = prepare_aif360_data(X_test_binned, y_test, protected_attribute)
        dataset_test_transf_final = OP_final.transform(dataset_test_orig, transform_Y=False)
        X_test_transf_final = pd.DataFrame(dataset_test_transf_final.features, columns=dataset_test_transf_final.feature_names)
        
        # Train final RandomForest on the transformed data
        best_model = RandomForestClassifier(**rf_params_best, random_state=SEED, n_jobs=1)
        best_model.fit(X_train_transf_final, y_train_transf_final)
        
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
        def opp_rf_predict_proba(X_binned_np):
            X_binned_df = pd.DataFrame(X_binned_np, columns=attributes)
            y_dummy = pd.Series(np.zeros(X_binned_df.shape[0]), name=label_col_name, index=X_binned_df.index)
            dataset_to_transform = prepare_aif360_data(X_binned_df, y_dummy, protected_attribute)
            
            dataset_transformed = OP_final.transform(dataset_to_transform, transform_Y=False)
            X_transformed = pd.DataFrame(dataset_transformed.features, columns=dataset_transformed.feature_names)
            
            return best_model.predict_proba(X_transformed)

        explanations = [
            explainer.explain_instance(
                data_row=X_test_binned.iloc[i].values, # Use discretized test
                predict_fn=opp_rf_predict_proba,
                num_features=len(attributes)
            ) for i in range(min(10, len(X_test_binned)))
        ]
        
        return df_final, best_model, final_best_params, final_f1, explanations
    else:
        print("\nERRO: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None


