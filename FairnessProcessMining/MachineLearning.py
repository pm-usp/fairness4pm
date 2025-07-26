# Libraries for data manipulation
import pandas as pd
import numpy as np
import random

#Libraries for machine learning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Library for hyperparameter optimization
import optuna

# Library for explainability
from lime.lime_tabular import LimeTabularExplainer

#Sementes
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def RandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, num_trials=50):
    """
    This function performs hyperparameter optimization for a RandomForestClassifier model using the Optuna library, trains the model with the best hyperparameters found, makes predictions on the training and testing data,
    and generates LIME explanations for the predictions made on the test set.
    
    Parameters:
    - X_train (pd.DataFrame): DataFrame containing the features of the training set.
    - y_train (pd.Series): Series containing the labels of the training set.
    - X_test (pd.DataFrame): DataFrame containing the features of the test set.
    - y_test (pd.Series): Series containing the labels of the test set.
    - df_train (pd.DataFrame): Original DataFrame of the training set, where the predictions will be added.
    - df_test (pd.DataFrame): Original DataFrame of the test set, where the predictions will be added.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Combined training and testing DataFrame with predictions and source information.
    - best_model (RandomForestClassifier): The best model trained using the best hyperparameters found.
    - best_params (dict): Dictionary containing the best hyperparameters found by optimization.
    - best_score (float): The best average F1 score found during optimization.
    - explanations (list): List of LIME explanations for the first 10 samples of the test set.
    """

    attributes = X_train.columns.tolist()
    
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
        
        # Evaluation with cross-validation and return of the average F1 score
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        f1 = cross_val_score(clf_rf, X_train, y_train, cv=skf, scoring='f1_weighted').mean()
        return f1   
    
    # Hyperparameter Optimization Using Optuna with Early Stopping (MedianPruner)
        #MedianPruner: stops training a set of hyperparameters if it is not performing well in intermediate steps of the optimization process
            #n_warmup_steps: hyperparameter runs are always completed completely, without pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)   
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5) 
    study  = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=7), 
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)
    
    # Getting the best hyperparameters and training the model with them
    best_params = study.best_params
    best_score = study.best_value
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Prediction on training and testing data
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Add the predictions and source to the training DataFrame
    df_train.loc[:, 'source'] = 'train'
    df_train.loc[:, 'prediction'] = y_pred_train
    df_test.loc[:, 'source'] = 'test'
    df_test.loc[:, 'prediction'] = y_pred_test
    
    # Combine the training and testing DataFrames
    df_final = pd.concat([df_train, df_test], ignore_index=True)
    
    explainer = LimeTabularExplainer(
    training_data=X_train.values, 
    feature_names=attributes, 
    class_names=['Classe 0', 'Classe 1'], 
    discretize_continuous=True,
    random_state=SEED
    
    )
    explanations = []
    for i in range(min(10, len(X_test))):
        instance_to_explain = X_test.iloc[i].values
        
        # AJUSTE: Usar 'attributes' (de X_train.columns) na lambda para consistência
        exp = explainer.explain_instance(
            data_row=instance_to_explain, 
            predict_fn=lambda x: best_model.predict_proba(pd.DataFrame(x, columns=attributes)), 
            num_features=len(attributes) # Usar len(attributes) para consistência
        )
        explanations.append(exp)
      
    # Return the final DataFrame, best model, best hyperparameters, best score, and LIME explanations
    return df_final, best_model, best_params, best_score, explanations

