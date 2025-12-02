# Libraries for data manipulation
import pandas as pd
import numpy as np
import random
import sys
import os

#Libraries for machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Libraries for fairness
directory = ''
sys.path.append(os.path.abspath(directory))
from AIF360.inprocessing.PrejudiceRemover import PrejudiceRemover

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


class SuppressPrints:
    """
    Context manager to temporarily suppress all output
    sent to sys.stdout (e.g. the print() function).
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class PrejudiceRemoverWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for AIF360's PrejudiceRemover to make it compatible
    with the scikit-learn interface and for use with LIME.
    """
    def __init__(self, sensitive_attr_name, class_attr_name='Target', eta=1.0, seed=None):
        self.sensitive_attr_name = sensitive_attr_name
        self.class_attr_name = class_attr_name 
        self.eta = eta
        self.seed = seed 
        
        self.model_ = None
        self.columns_ = None
        self.classes_ = None
        self.scaler_ = None 

    def _set_seeds_if_needed(self):
        if self.seed is not None:
            np.random.seed(self.seed)

    def fit(self, X, y):
        self._set_seeds_if_needed()
        self.columns_ = X.columns.tolist()
        self.classes_ = np.unique(y)

        # PrejudiceRemover may be sensitive to feature scale
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)

        y_named = y.copy()
        if y_named.name is None:
            y_named.name = self.class_attr_name 
            
        aif360_dataset = prepare_aif360_data(X_scaled_df, y_named, self.sensitive_attr_name)
        
        self.model_ = PrejudiceRemover(eta=self.eta, 
                                       sensitive_attr=self.sensitive_attr_name,
                                       class_attr=self.class_attr_name) 
        
        self.model_.fit(aif360_dataset)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Untrained model. Call fit first.")
    
        # Check if the input is a DataFrame to preserve the original index
        # If it is a NumPy array (from LIME), the index will be None (pandas default)
        original_index = X.index if isinstance(X, pd.DataFrame) else None
        
        X_scaled = self.scaler_.transform(X)
        # Use the original index (if it exists) and the columns saved during 'fit'
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.columns_, index=original_index)
        
        y_dummy = pd.Series(np.zeros(X_scaled_df.shape[0]), name=self.class_attr_name, index=X_scaled_df.index)
        aif360_dataset_pred = prepare_aif360_data(X_scaled_df, y_dummy, self.sensitive_attr_name)
        
        return self.model_.predict(aif360_dataset_pred).labels.ravel()


    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Untrained model. Call fit first.")

        original_index = X.index if isinstance(X, pd.DataFrame) else None
        X_scaled = self.scaler_.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.columns_, index=original_index)
    
        y_dummy = pd.Series(np.zeros(X_scaled_df.shape[0]), name=self.class_attr_name, index=X_scaled_df.index)
        aif360_dataset_pred = prepare_aif360_data(X_scaled_df, y_dummy, self.sensitive_attr_name)
        
        pred_results = self.model_.predict(aif360_dataset_pred)
        scores = pred_results.scores.ravel()
        proba_favorable = 1 / (1 + np.exp(-scores)) 
        
        proba = np.vstack([1 - proba_favorable, proba_favorable]).T
        
        # Verification
        if not np.allclose(proba.sum(axis=1), 1.0):
            print(f"Warning: Probabilities in PrejudiceRemoverWrapper do not sum to 1."
            f"First 5 sums: {proba.sum(axis=1)[:5]}. Normalizing...")
            proba_sum = proba.sum(axis=1, keepdims=True)
            proba_sum[proba_sum == 0] = 1 # Avoid division by zero if both tests are 0
            proba = proba / proba_sum
    
        return proba


def InPrejudiceRemoverOptuna(X_train, y_train, X_test, y_test, 
                             df_train, df_test, 
                             protected_attribute, num_trials=50):
    """
    This function implements the AIF360 PrejudiceRemover model to mitigate bias during model training. 
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
        eta_val = trial.suggest_float('eta', 1e-2, 100.0, log=True)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
                      
            y_fold_train_named = y_fold_train.copy()
            if y_fold_train_named.name is None: y_fold_train_named.name = label_col_name
            if not y_fold_train_named.index.equals(y_fold_train.index):
                 y_fold_train_named = pd.Series(y_fold_train_named.values, index=y_fold_train.index, name=y_fold_train_named.name)
            
            model = PrejudiceRemoverWrapper(
                sensitive_attr_name=protected_attribute,
                class_attr_name=label_col_name,
                eta=eta_val,
                seed=SEED
            )
            
            y_pred_val = np.array([]) 

            try:
                # The wrapper handles data preparation internally
                with SuppressPrints(): 
                    model.fit(X_fold_train, y_fold_train_named) 
                y_pred_val = model.predict(X_fold_val)
            except Exception as e:
                pass 
            
            if y_pred_val.shape[0] != X_fold_val.shape[0]:
                f1_scores.append(0.0)
                di_scores.append(0.0)
                continue 
            
            y_fold_val_named = y_fold_val.copy()
            if y_fold_val_named.name is None: y_fold_val_named.name = label_col_name
            if not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val_named.values, index=X_fold_val.index, name=y_fold_val_named.name)

            f1 = f1_score(y_fold_val_named, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
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
            # fig_pareto.write_html("InPR_pareto_front.html") # Para salvar
            
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
        
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
        
        print("\n--- Final Configuration Selected By Main Script ---")
        if final_f1 is not None:
            print(f"F1-score (CV of the selected trial): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV of selected trial): {final_di:.4f}")
        print(f"Best Hyperparameters: {final_best_params_to_return}")

        # --- Final Training ---
        # Instantiate the model with the best parameters
        best_model = PrejudiceRemoverWrapper(
            sensitive_attr_name=protected_attribute,
            class_attr_name=label_col_name,
            **params_for_model, 
            seed=SEED
        )
        
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        
        y_pred_train = np.array([])
        y_pred_test = np.array([])

        try:
            with SuppressPrints():
                best_model.fit(X_train, y_train) 
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
        except Exception as e:
            print(f"CRITICAL ERROR in training/prediction of the final model: {e}")
            return None, None, None, None, None

        # Final sanity check
        if y_pred_train.shape[0] != len(X_train) or y_pred_test.shape[0] != len(X_test):
            print("\CRITICAL ERROR: Final model failed to create predictions.")
            return None, None, None, None, None
        
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
        for i in range(min(10, len(X_test))):
            instance_to_explain = X_test.iloc[i].values
            
            exp = explainer.explain_instance(
                data_row=instance_to_explain, 
                predict_fn=best_model.predict_proba, 
                num_features=len(attributes) 
            )
            explanations.append(exp)
               
        return df_final, best_model, final_best_params_to_return, final_f1, explanations
    else:
        print("\nERRO: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None