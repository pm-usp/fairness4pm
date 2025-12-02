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
from AIF360.inprocessing.AdversarialDebiasing import AdversarialDebiasing

from FairnessProcessMining.AIF360.auxiliary import prepare_aif360_data
from FairnessProcessMining.AIF360.auxiliary import select_best_hyperparameters_trial

# TensorFlow (required for AdversarialDebiasing)
import tensorflow.compat.v1 as tf

# Library for hyperparameter optimization
import optuna

# Library for explainability
from lime.lime_tabular import LimeTabularExplainer

# Library for fairness metrics
from FairnessProcessMining import Metrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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

  

def InAdversarialDebiasingOptuna(X_train, y_train, X_test, y_test, 
                                 df_train, df_test, 
                                 protected_attribute, num_trials=50):
    """
    This function implements the AIF360 Adversarial Debiasing model to mitigate bias during model training. 
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

    # LIME Explainer is initialized with ORIGINAL (unscaled) data
    # The wrapper predict_proba function will handle scaling internally
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
        weight = trial.suggest_float('adversary_loss_weight', 0.05, 1.0, step=0.05) 
      
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) 
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
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["Average CV F1-score", "Average CV Fairness Score"])
            fig_pareto.show()
            # To save: fig_pareto.write_html("AdversarialDebiasing_pareto_front.html")

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
        
        # --- Final Training ---
        # Instantiate the model with the best parameters
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
        
        best_model.close_session() 
        
        return df_final, best_model, final_best_params_to_return, best_score_f1_cv, explanations
    else:
        print("\nERRO: Unable to select a trial with the defined criteria.")
        return None, None, None, None, None
