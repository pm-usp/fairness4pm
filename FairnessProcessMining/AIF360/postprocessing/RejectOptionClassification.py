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
from AIF360.postprocessing.RejectOptionClassification import RejectOptionClassification

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


def PostROCRandomForestOptuna(X_train, y_train, X_test, y_test, 
                              df_train, df_test, 
                              protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    AIF360 Reject Option Classification to adjust predictions.
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

        # Hyperparameter to RejectOptionClassification
        roc_metric_name = trial.suggest_categorical(
            'roc_metric_name', 
            ["Statistical parity difference", 
             "Average odds difference", 
             "Equal opportunity difference"]
        )
        
        roc_metric_ub = trial.suggest_float('roc_metric_ub', 0.0, 0.1)
        roc_metric_lb = trial.suggest_float('roc_metric_lb', -0.1, 0.0)

        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            criterion=criterion, n_jobs=1, random_state=SEED
        )

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) # Usando k=3 para robustez
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            
            # ROC needs the PROBABILITIES of the base model
            y_val_pred_proba = clf_rf.predict_proba(X_fold_val)
            scores_favorable_class = y_val_pred_proba[:, 1]
            y_val_pred_labels = (scores_favorable_class >= 0.5).astype(int)

            # Prepare AIF360 datasets with SCORES
            y_fold_val_named = y_fold_val.copy() 
            if y_fold_val_named.name is None: y_fold_val_named.name = label_col_name
            if not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val.values, index=X_fold_val.index, name=y_fold_val_named.name)
            
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val_named, protected_attribute)
            bd_pred = bd_true.copy(deepcopy=True)
            bd_pred.labels = y_val_pred_labels.reshape(-1, 1)
            bd_pred.scores = scores_favorable_class.reshape(-1, 1)
            
            f1_fold, di_fold = 0.0, 0.0

            try:
                roc = RejectOptionClassification(
                    unprivileged_groups=[{protected_attribute: 1}],
                    privileged_groups=[{protected_attribute: 0}],
                    low_class_thresh=0.01, 
                    high_class_thresh=0.99,
                    num_class_thresh=10, 
                    num_ROC_margin=10,
                    metric_name=roc_metric_name,
                    metric_ub=roc_metric_ub,
                    metric_lb=roc_metric_lb
                )
                roc.fit(bd_true, bd_pred)
                bd_pred_post = roc.predict(bd_pred)
                y_post_val = bd_pred_post.labels.ravel()

                f1_fold = f1_score(y_fold_val_named, y_post_val, average='weighted')
                
                df_val_metric = X_fold_val.copy()
                df_val_metric[label_col_name] = y_fold_val_named.values
                df_val_metric['prediction'] = y_post_val
                
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
    if len(study.trials) > 0:
        try:
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Average CV F1-score", "Average CV Fairness Score"])
            fig_pareto.show()
            # fig_pareto.write_html("PostROC_pareto_front.html")

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
        
        # Separate ROC and RandomForest parameters from the full dictionary
        roc_params_best = {k.replace('roc_', ''): v for k, v in final_best_params_to_return.items() if k.startswith('roc_')}
        rf_params_best = {k: v for k, v in final_best_params_to_return.items() if not k.startswith('roc_')}
        
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best  ROC (CV): {roc_params_best}") 
        print(f"Best Hyperparameters (RandomForest): {rf_params_best}")
                
        # --- Final Training ---    
        best_model = RandomForestClassifier(**rf_params_best, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_pred_proba  = best_model.predict_proba(X_test)[:, 1]
        y_train_pred_labels = (y_train_pred_proba >= 0.5).astype(int)
        y_test_pred_labels  = (y_test_pred_proba >= 0.5).astype(int)

        bd_train_true = prepare_aif360_data(X_train, y_train, protected_attribute)
        bd_train_pred = bd_train_true.copy(deepcopy=True)
        bd_train_pred.labels = y_train_pred_labels.reshape(-1, 1)
        bd_train_pred.scores = y_train_pred_proba.reshape(-1, 1)
        
        bd_test_orig = prepare_aif360_data(X_test, y_test, protected_attribute)
        bd_test_pred = bd_test_orig.copy(deepcopy=True)
        bd_test_pred.labels = y_test_pred_labels.reshape(-1, 1)
        bd_test_pred.scores = y_test_pred_proba.reshape(-1, 1)
        
        roc_final = RejectOptionClassification(
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}],
            low_class_thresh=0.01, 
            high_class_thresh=0.99,
            num_class_thresh=100, 
            num_ROC_margin=50,
            **roc_params_best   # 'metric_name', 'metric_ub', 'metric_lb'
        )
        
        y_train_pred_final = y_train_pred_labels
        y_test_pred_final = y_test_pred_labels
        lime_predict_fn = lambda X_lime_input_df: best_model.predict_proba(pd.DataFrame(X_lime_input_df, columns=attributes))

        try:
            roc_final.fit(bd_train_true, bd_train_pred)
            
            y_train_pred_post = roc_final.predict(bd_train_pred).labels.ravel()
            y_test_pred_post  = roc_final.predict(bd_test_pred).labels.ravel()
            
            y_train_pred_final = y_train_pred_post
            y_test_pred_final = y_test_pred_post   

            def post_processed_predict_proba(X_lime_input_np):
                X_lime_df = pd.DataFrame(X_lime_input_np, columns=attributes)
                probs_raw = best_model.predict_proba(X_lime_df)
                scores_favorable = probs_raw[:, 1]
                labels_raw = (scores_favorable >= 0.5).astype(int)
                
                y_dummy_series = pd.Series(labels_raw, index=X_lime_df.index, name=label_col_name)
                bd_lime = prepare_aif360_data(X_lime_df, y_dummy_series, protected_attribute)
                bd_lime.scores = scores_favorable.reshape(-1, 1)
                
                bd_lime_post = roc_final.predict(bd_lime)
                final_labels = bd_lime_post.labels.ravel()
                return np.array([[1-p, p] for p in final_labels])
            
            lime_predict_fn = post_processed_predict_proba
        except Exception as e:
            print(f"WARNING: Solver error when fitting ROC on the full dataset ({e}). Using raw predictions.")
        
        df_train['source'] = 'train'
        df_train['prediction'] = y_train_pred_final
        df_test['source'] = 'test'
        df_test['prediction'] = y_test_pred_final
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        explanations = [
            explainer.explain_instance(
                data_row=X_test.iloc[i].values,
                predict_fn=lime_predict_fn,
                num_features=len(attributes) 
            ) for i in range(min(10, len(X_test)))
        ]
        
        return df_final, best_model, final_best_params_to_return, final_f1, explanations
    else:
        print("\nERRO: Não foi possível selecionar um trial com os critérios definidos...")
        return None, None, None, None, None


