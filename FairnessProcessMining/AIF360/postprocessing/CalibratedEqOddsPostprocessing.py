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
from AIF360.postprocessing.CalibratedEqOddsPostprocessing import CalibratedEqOddsPostprocessing

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



def PostCalibratedEOddsRandomFlorestOptuna(X_train, y_train, X_test, y_test, 
                                           df_train, df_test, 
                                           protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    AIF360 Calibrated Equalized Odds Postprocessing to adjust predictions.
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
             
        cost_constraint = trial.suggest_categorical("cost_constraint", ["fnr", "fpr", "weighted"])
        
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
            # Pareto Frontier Chart
            # Objective names are inferred from the order returned in 'objective'
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Average CV F1-score", "Average CV Fairness Score"])
            fig_pareto.show()
            # To save: fig_pareto.write_html("PostCalibEOdds_pareto_front.html") # Nome de arquivo específico

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
        
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best cost_constraint (CV): {cost_constraint_best}") 
        print(f"Best Hyperparameters (RandomForest): {rf_params_best}")

        # --- Final Training ---    
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
                print(f"WARNING: Solver error when fitting CalibratedEqOdds on the full dataset ({e}). Using raw predictions.")
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

        # Generate LIME explanations 
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
        
        return df_final, best_model, final_best_params_to_return, final_f1, explanations 
    else:
        print("\nERROR: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None