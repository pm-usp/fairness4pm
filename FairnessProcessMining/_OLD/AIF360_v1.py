# Libraries for data manipulation
import pandas as pd
import numpy as np
import random
from typing import Optional, List, Dict, Any
import sys
import os

#Libraries for machine learning
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
    Seleciona o melhor trial de um estudo Optuna com base em uma lógica customizada
    para F1-score e Disparate Impact (DI).

    A lógica de seleção é hierárquica:
    1. Tier 1: Prioriza trials com DI dentro do intervalo [di_tier1_min, di_tier1_max],
       selecionando aquele com o maior F1-score.
    2. Tier 2 (Fallback): Se nenhum trial atender ao Tier 1, busca trials com DI
       no intervalo [di_tier2_min, di_tier2_max (exclusive)), ordenando primeiro
       pelo DI (descendente, para o mais próximo de di_tier2_max) e, em seguida,
       pelo F1-score (descendente) como critério de desempate.
    3. Fallback Geral: Se nenhum trial atender ao Tier 1 ou Tier 2, seleciona o
       trial com o maior F1-score geral entre todos os trials válidos.

    Args:
        study: O objeto optuna.study.Study após a execução de study.optimize().
        di_tier1_min: Limite inferior do DI para o Critério Tier 1.
        di_tier1_max: Limite superior do DI para o Critério Tier 1.
        di_tier2_min: Limite inferior do DI para o Critério Tier 2.
        di_tier2_max: Limite superior (exclusive) do DI para o Critério Tier 2.

    Returns:
        Um objeto optuna.trial.FrozenTrial representando o melhor trial selecionado,
        ou None se nenhum trial adequado for encontrado.
    """

    all_completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not all_completed_trials:
        print("LOG:select_best_trial: Erro - Nenhum trial do Optuna foi concluído.")
        return None

    # Preparar uma lista de dicionários para facilitar a ordenação e acesso
    # Considerar apenas trials com métricas 'avg_f1' e 'avg_di' válidas e DI finito
    valid_trials_info: List[Dict[str, Any]] = []
    for t in all_completed_trials:
        f1 = t.user_attrs.get("avg_f1")
        di = t.user_attrs.get("avg_di")
        if f1 is not None and di is not None and np.isfinite(di):
            valid_trials_info.append({"trial_obj": t, "f1": f1, "di": di})

    if not valid_trials_info:
        print(
            "LOG:select_best_trial: Erro - Nenhum trial concluído possui as métricas "
            "'avg_f1' e 'avg_di' válidas e com DI finito."
        )
        return None

    selected_trial_obj: Optional[optuna.trial.FrozenTrial] = None

    # Critério Tier 1
    tier1_candidates = [
        info for info in valid_trials_info if di_tier1_min <= info["di"] <= di_tier1_max
    ]
    if tier1_candidates:
        tier1_candidates.sort(key=lambda x: x["f1"], reverse=True)
        selected_trial_obj = tier1_candidates[0]["trial_obj"]
        print(
            f"LOG:select_best_trial: Selecionado via Critério Tier 1 "
            f"(DI em [{di_tier1_min:.2f}, {di_tier1_max:.2f}], maior F1). "
            f"F1: {tier1_candidates[0]['f1']:.4f}, DI: {tier1_candidates[0]['di']:.4f}"
        )
    else:
        # Critério Tier 2
        tier2_candidates = [
            info for info in valid_trials_info if di_tier2_min <= info["di"] < di_tier2_max
        ]
        if tier2_candidates:
            tier2_candidates.sort(key=lambda x: (x["di"], x["f1"]), reverse=True)
            selected_trial_obj = tier2_candidates[0]["trial_obj"]
            print(
                f"LOG:select_best_trial: Selecionado via Critério Tier 2 "
                f"(DI em [{di_tier2_min:.2f}, {di_tier2_max:.2f}), maior DI, depois maior F1). "
                f"F1: {tier2_candidates[0]['f1']:.4f}, DI: {tier2_candidates[0]['di']:.4f}"
            )

    # Fallback Geral
    if not selected_trial_obj:
        print(
            "LOG:select_best_trial: AVISO - Nenhum trial atendeu aos Critérios Tier 1 ou Tier 2. "
            "Aplicando fallback: maior F1 geral."
        )
        # valid_trials_info já contém todos os trials com métricas válidas
        if valid_trials_info: # Deve ser verdadeiro se o primeiro check de valid_trials_info passou
            valid_trials_info.sort(key=lambda x: x["f1"], reverse=True)
            selected_trial_obj = valid_trials_info[0]["trial_obj"]
            print(
                f"LOG:select_best_trial: Selecionado via Fallback Geral (maior F1). "
                f"F1: {valid_trials_info[0]['f1']:.4f}, DI: {valid_trials_info[0]['di']:.4f}"
            )

    if not selected_trial_obj:
        # Este caso só deve ocorrer se valid_trials_info estava vazio inicialmente,
        # o que já é tratado no início.
        print("LOG:select_best_trial: ERRO CRÍTICO - Não foi possível selecionar nenhum trial.")

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
    
    # Create a list of feature names from the training DataFrame
    attributes = X_train.columns.tolist()

    # Create datasets for the AIF360 library
    dataset_train, dataset_test = AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute)

    # Apply the Reweighing technique to the training dataset
    # Create reweighing object
    RW = Reweighing(unprivileged_groups=[{protected_attribute: 1}],
                    privileged_groups=[{protected_attribute: 0}])
    
    # transform training data
    dataset_transf_train = RW.fit_transform(dataset_train) 
    
    # Convert back to DataFrame
    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name=y_train.name, index=X_train_transf.index)
    weights = pd.Series(dataset_transf_train.instance_weights, index=X_train_transf.index)

    # Optimization function for Optuna
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Number of estimators
        max_depth = trial.suggest_int('max_depth', 5, 20)  # Maximum depth
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)  # Minimum number of samples per sheet
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        # max_samples condicionalmente
        if bootstrap_enabled:
            # max_samples só é aplicável se bootstrap=True
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05) # step opcional para granularidade
        else:
            # Se bootstrap=False, max_samples deve ser None
            max_samples_val = None
    
        
        # Instantiating the RandomForest model with the suggested hyperparameters
        clf_rf = RandomForestClassifier(
             n_estimators=n_estimators,
             max_depth=max_depth,
             min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf,
             max_features=max_features,
             bootstrap=bootstrap_enabled,
             class_weight=class_weight,
             max_samples=max_samples_val,
             n_jobs=1,
             random_state=SEED,
         )
        
        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
        
           
        for train_index, val_index in skf.split(X_train_transf, y_train_transf):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train_transf.iloc[train_index], X_train_transf.iloc[val_index]
            y_fold_train, y_fold_val = y_train_transf.iloc[train_index], y_train_transf.iloc[val_index]
            weights_fold_train = weights[train_index]
    
            # Fit the model on the fold with instance weights
            clf_rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
    
            # Predict and compute weighted F1
            y_pred_val = clf_rf.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
    
            # Prepare DataFrame for fairness metric computation
            df_val = pd.DataFrame(X_fold_val, columns=attributes)
            df_val['Target'] = y_fold_val.values
            df_val['prediction'] = y_pred_val
    
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
    
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
            
        # Convert disparate impact into a fairness score in [0,1]
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        # Salvar os valores brutos de avg_f1 e avg_di para seleção customizada posterior
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
    
        # Return both objectives to maximize
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
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params
        # Acessar as métricas salvas para o log final
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---")
        if final_f1 is not None:
            print(f"F1-score (do trial selecionado): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {best_params}")

        # Train final model on full transformed training data
        best_model = RandomForestClassifier(**best_params, random_state=SEED)
        best_model.fit(X_train_transf, y_train_transf, sample_weight=weights)
    
        # Prediction on training and testing data
        y_pred_train = best_model.predict(X_train_transf)
        y_pred_test = best_model.predict(X_test)
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)
    
        # Generate LIME explanations for the first 10 test instances
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=attributes,
            class_names=[y_train.name],
            discretize_continuous=True,
            random_state=SEED
        )
        explanations = []
        for i in range(min(10, X_test.shape[0])):
            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                lambda x: best_model.predict_proba(pd.DataFrame(x, columns=attributes)),
                num_features=X_test.shape[1]
            )
            explanations.append(exp)
    
        # Return the final DataFrame, best model, best hyperparameters, best score, and LIME explanations
        return df_final, best_model, best_params, final_f1, explanations

    else:
        print(
            "\nERRO no script principal: Não foi possível selecionar um trial com os critérios definidos. "
            "O treinamento otimizado não pode prosseguir."
        )
        return None, None, None, None, None
   

def PreDisparateImpactRemoverRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    Applies the nonsense impact removal (DIR) technique to the training data and trains a RandomForest model,
    optimizing its hyperparameters with Optuna and using cross-validation. 
    The function also calculates the nonsense impact and returns LIME explanations for the model's predictions.
    
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
    
    # Create a list of feature names from the training DataFrame
    attributes = X_train.columns.tolist()
    
    # Create datasets for the AIF360 library
    dataset_train, dataset_test = AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute)

    # Apply Disparate Impact Remover technique only on the training dataset
    DIR = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=protected_attribute)  # Complete repair
    dataset_transf_train = DIR.fit_transform(dataset_train)

    # Convert back to DataFrame
    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name=y_train.name, index=X_train_transf.index)

    # Define Optuna objective for multi-objective optimization
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Number of estimators
        max_depth = trial.suggest_int('max_depth', 5, 20)  # Maximum depth
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)  # Minimum number of samples per sheet
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        # max_samples condicionalmente
        if bootstrap_enabled:
            # max_samples só é aplicável se bootstrap=True
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05) # step opcional para granularidade
        else:
            # Se bootstrap=False, max_samples deve ser None
            max_samples_val = None
        
        # Instantiating the RandomForest model with the suggested hyperparameters
        clf_rf = RandomForestClassifier(
             n_estimators=n_estimators,
             max_depth=max_depth,
             min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf,
             max_features=max_features,
             bootstrap=bootstrap_enabled,
             class_weight=class_weight,
             max_samples=max_samples_val,
             n_jobs=1,
             random_state=SEED,
         )
        
        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train_transf, y_train_transf):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train_transf.iloc[train_index], X_train_transf.iloc[val_index]
            y_fold_train, y_fold_val = y_train_transf.iloc[train_index], y_train_transf.iloc[val_index]
    
            # Fit the model on the fold with instance weights
            clf_rf.fit(X_fold_train, y_fold_train)
    
            # Predict and compute weighted F1
            y_pred_val = clf_rf.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
    
            # Prepare DataFrame for fairness metric computation
            df_val = pd.DataFrame(X_fold_val, columns=attributes)
            df_val['Target'] = y_fold_val.values
            df_val['prediction'] = y_pred_val
    
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
    
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
            
        # Convert disparate impact into a fairness score in [0,1]
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        # Salvar os valores brutos de avg_f1 e avg_di para seleção customizada posterior
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
    
        # Return both objectives to maximize
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
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params
        # Acessar as métricas salvas para o log final
        final_f1 = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---")
        if final_f1 is not None:
            print(f"F1-score (do trial selecionado): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {best_params}")

        # Train final model on full transformed training data
        best_model = RandomForestClassifier(**best_params, random_state=SEED)
        best_model.fit(X_train_transf, y_train_transf)
    
        # Prediction on training and testing data
        y_pred_train = best_model.predict(X_train_transf)
        y_pred_test = best_model.predict(X_test)
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)
    
        # Generate LIME explanations for the first 10 test instances
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=attributes,
            class_names=[y_train.name],
            discretize_continuous=True,
            random_state=SEED
        )
        explanations = []
        for i in range(min(10, X_test.shape[0])):
            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                lambda x: best_model.predict_proba(pd.DataFrame(x, columns=attributes)),
                num_features=X_test.shape[1]
            )
            explanations.append(exp)
    
        # Return the final DataFrame, best model, best hyperparameters, best score, and LIME explanations
        return df_final, best_model, best_params, final_f1, explanations

    else:
        print(
            "\nERRO no script principal: Não foi possível selecionar um trial com os critérios definidos. "
            "O treinamento otimizado não pode prosseguir."
        )
        return None, None, None, None, None
    


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                       In-processing Models                               """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Required for compatibility with TensorFlow 1.x and AIF360
tf.compat.v1.disable_v2_behavior()

class SuppressPrints: # Defina esta classe uma vez no seu script
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class AdversarialDebiasingWrapper(BaseEstimator, ClassifierMixin):
    """
    The `AdversarialDebiasingWrapper` class implements an adversarial learning model to debias 
    protected attributes during training. It uses the AIF360 library and TensorFlow 1.x to create 
    a classifier that takes bias into account and adjusts the model to balance predictions based 
    on privileged and unprivileged attributes. The class also provides methods for performing 
    predictions and prediction probabilities.
    """

    def __init__(self, protected_attribute_name, privileged_groups, unprivileged_groups, scope_name='debiased_classifier',
                 debias=True, num_epochs=100, classifier_num_hidden_units=100, adversary_loss_weight=0.1, seed=SEED):
        self.protected_attribute_name = protected_attribute_name
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.scope_name = scope_name
        self.debias = debias
        self.num_epochs = num_epochs
        self.classifier_num_hidden_units = classifier_num_hidden_units
        self.adversary_loss_weight = adversary_loss_weight
        self.seed = seed
        self.sess = None
        self.model = None
        self.columns = None  # Inicializa columns como None
        self._set_seeds()

    def _set_seeds(self):
        """
        Sets seeds to ensure reproducibility of training and evaluation results.
        """
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

    def fit(self, dataset):
        """
        Trains the AdversarialDebiasing model with the provided dataset.
        The dataset is preprocessed using the AIF360 library, and the adversarial model is fine-tuned 
        to remove bias during the training process.
        """
        tf.compat.v1.reset_default_graph()
        self._set_seeds()  # Reset seeds when resetting graph
        self.sess = tf.compat.v1.Session()
        self.model = AdversarialDebiasing(privileged_groups=self.privileged_groups,
                                          unprivileged_groups=self.unprivileged_groups,
                                          scope_name=self.scope_name,
                                          debias=self.debias,
                                          num_epochs=self.num_epochs,
                                          classifier_num_hidden_units=self.classifier_num_hidden_units,
                                          adversary_loss_weight=self.adversary_loss_weight,
                                          sess=self.sess,
                                          seed=self.seed)
        self.model.fit(dataset)
        self.columns = dataset.feature_names  # Define columns after fitting the model
        return self

    def predict(self, dataset_input): # Renomeado para clarear o escopo
        """
        Performs label (class) predictions based on the provided input dataset.
        """
        # Garante que o input seja um dataset AIF360
        if isinstance(dataset_input, np.ndarray):
            # Se for ndarray, X são as features, y são dummies
            X_df = pd.DataFrame(dataset_input, columns=self.columns)
            y_dummy = pd.Series(np.zeros(len(dataset_input)), name='Target') # Nome consistente com prepare_aif360_data
            aif360_dataset = prepare_aif360_data(X_df, y_dummy, self.protected_attribute_name)
        elif isinstance(dataset_input, pd.DataFrame):
            # Se for DataFrame, X são as features, y são dummies
            # Assume que o DataFrame 'dataset_input' contém apenas as features com os nomes corretos
            X_df = dataset_input 
            y_dummy = pd.Series(np.zeros(len(X_df)), name='Target')
            aif360_dataset = prepare_aif360_data(X_df, y_dummy, self.protected_attribute_name)
        elif isinstance(dataset_input, StandardDataset) or isinstance(dataset_input, BinaryLabelDataset):
            # Se já for um dataset AIF360, usa diretamente
            aif360_dataset = dataset_input
        else:
            raise TypeError(f"Input dataset must be a NumPy array, Pandas DataFrame, or AIF360 Dataset object, got {type(dataset_input)}")

        predictions = self.model.predict(aif360_dataset)
        return predictions.labels.ravel()

    def predict_proba(self, dataset_input): # Renomeado para clarear o escopo
        """
        Performs probability predictions for the given dataset.
        Returns the predicted probabilities for each class (e.g. [prob_class_0, prob_class_1]).
        """
        # Garante que o input seja um dataset AIF360
        if isinstance(dataset_input, np.ndarray):
            X_df = pd.DataFrame(dataset_input, columns=self.columns)
            y_dummy = pd.Series(np.zeros(len(dataset_input)), name='Target')
            aif360_dataset = prepare_aif360_data(X_df, y_dummy, self.protected_attribute_name)
        elif isinstance(dataset_input, pd.DataFrame):
            X_df = dataset_input
            y_dummy = pd.Series(np.zeros(len(X_df)), name='Target')
            aif360_dataset = prepare_aif360_data(X_df, y_dummy, self.protected_attribute_name)
        elif isinstance(dataset_input, StandardDataset) or isinstance(dataset_input, BinaryLabelDataset):
            aif360_dataset = dataset_input
        else:
            raise TypeError(f"Input dataset must be a NumPy array, Pandas DataFrame, or AIF360 Dataset object, got {type(dataset_input)}")

        predictions = self.model.predict(aif360_dataset) # self.model é o AdversarialDebiasing da AIF360
        # 'scores' em AdversarialDebiasing.predict() geralmente são as probabilidades da classe favorável
        favorable_class_probs = predictions.scores.ravel()
        proba = np.vstack([1 - favorable_class_probs, favorable_class_probs]).T
        
        # Check if the probabilities sum to 1 (esta parte parece correta)
        # print("Probas:", proba[:5], "Sums:", proba.sum(axis=1)[:5]) 
        if not np.allclose(proba.sum(axis=1), 1.0):
            # Pode acontecer se scores não estiverem no intervalo [0,1]
            # Normalizar se necessário ou investigar a saída de scores
            # Exemplo de normalização (cuidado, isso mascara problemas se scores não forem probabilidades)
            # proba = proba / proba.sum(axis=1, keepdims=True)
            print(f"Warning: Probabilities do not sum to 1. First 5 sums: {proba.sum(axis=1)[:5]}")
            # raise ValueError("Prediction probabilities do not sum to 1.") # Pode ser muito estrito se houver pequenos erros de float
        return proba

    def close_session(self):
        """
        Ends the TensorFlow session to free up memory resources.
        """
        if self.sess is not None:
            self.sess.close()


def prepare_aif360_data(X_df_features, y_series_labels, protected_attribute_name):
    """
    Prepares input data in the AIF360 StandardDataset format.
    """
    df_combined = X_df_features.copy()
    # Garante que y_series_labels tenha o nome correto
    current_label_name = y_series_labels.name if y_series_labels.name is not None else 'Target'
    df_combined[current_label_name] = y_series_labels
    
    return StandardDataset(df_combined, 
                           label_name=current_label_name, 
                           favorable_classes=[1], # Ou a classe que for favorável no seu contexto
                           protected_attribute_names=[protected_attribute_name], 
                           # Assume que o valor 0 no atributo protegido é privilegiado
                           # Isso deve ser consistente com privileged_groups no AdversarialDebiasingWrapper
                           privileged_classes=[[0]])


def InAdversarialDebiasingOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    Implements the Adversarial Debiasing model with AIF360 and TensorFlow
    to mitigate bias towards protected attributes during model training.
    The model's hyperparameters are optimized using Optuna, balancing predictive
    performance (F1-score) and fairness (Disparate Impact close to 1).
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - num_trials (int): Number of attempts for hyperparameter optimization with Optuna (default: 50).
    
    Returns:
    - df_final (pd.DataFrame): Final DataFrame with the combined training and testing predictions.
    - best_model: Best trained model after optimization.
    - best_params (dict): Optimized hyperparameters.
    - best_score (float): Best score resulting from optimization.
    - explanations (list): LIME explanations for the predictions.
    """
    
    # Create a list of feature names from the training DataFrame
    attributes = X_train.columns.tolist()
    
    # Prepare LIME explainer once for consistency
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['0', '1'],
        discretize_continuous=True,
        random_state=SEED
    )

    def objective(trial):
        """
        Objective function for optimizing hyperparameters of the adversarial model.
        Uses the Optuna framework to search for the best parameters, balancing the F1-score and the disparity impact (DI).
        """
        num_epochs = trial.suggest_categorical('num_epochs', [50, 100, 200])
        units = trial.suggest_categorical('classifier_num_hidden_units', [50, 100, 200])
        weight = trial.suggest_float('adversary_loss_weight', 0.1, 2.0)  # Tests values ​​in the range between 0.1 and 2.0

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
            # Instancia o model
            model = AdversarialDebiasingWrapper(
                protected_attribute_name=protected_attribute,
                privileged_groups=[{protected_attribute: 0}],
                unprivileged_groups=[{protected_attribute: 1}],
                num_epochs=num_epochs,
                classifier_num_hidden_units=units,
                adversary_loss_weight=weight,
                seed=SEED
            )
    
            # Prepara dados AIF360
            dataset_train = prepare_aif360_data(X_fold_train, y_fold_train, protected_attribute)
            
            # Suprimir prints durante o fit do fold
            #print(f"INFO: Training fold for trial {trial.number} - Suppressing AIF360 prints...") # Log opcional
            with SuppressPrints():
                #Treina e avalia
                model.fit(dataset_train)
            #print(f"INFO: Fold training complete for trial {trial.number}.") # Log opcional
            
            # predição no fold de validação
            dataset_val = prepare_aif360_data(X_fold_val, y_fold_val, protected_attribute)
            y_pred_val = model.predict(dataset_val)
            model.close_session()

            # Compute F1 score
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            # Compute disparate impact
            df_val = pd.DataFrame(X_fold_val, columns=attributes)
            df_val['Target'] = y_fold_val.values
            df_val['prediction'] = y_pred_val
            
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, 
                                              privileged_group=0, unprivileged_group=1,
                                              prediction='prediction')
            di_scores.append(di)
                   
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
        
        # Salvar os valores brutos de avg_f1 e avg_di para seleção customizada posterior
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
            
        # Convert disparate impact into a fairness score in [0,1]
        fairness_score = 1.0 - abs(avg_di - 1.0)
    
        # Return both objectives to maximize
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

    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---") 
    best_trial_found = select_best_hyperparameters_trial(study)
    
    # O restante do código já usa best_trial_found, o que está correto.
    # Apenas garanta que a atribuição de best_params e best_score venha de best_trial_found
    if best_trial_found:
        best_params = best_trial_found.params
        best_score = best_trial_found.user_attrs.get("avg_f1") # F1 do trial selecionado
        final_di = best_trial_found.user_attrs.get("avg_di") # Adicionar para log
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---") # Adicionar
        if best_score is not None: # best_score é o F1
            print(f"F1-score (do trial selecionado): {best_score:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {best_params}")       
    
        # Train final model with best params
        best_model = AdversarialDebiasingWrapper(
            protected_attribute_name=protected_attribute,
            privileged_groups=[{protected_attribute: 0}],
            unprivileged_groups=[{protected_attribute: 1}],
            **best_params,  # Unpack parameters
            seed=SEED
        )
    
        dataset_train = prepare_aif360_data(X_train, y_train, protected_attribute)
        dataset_test = prepare_aif360_data(X_test, y_test, protected_attribute)
        
        # Suprimir prints durante o fit do modelo final
        #print("INFO: Training final best model - Suppressing AIF360 prints...") # Log opcional seu
        with SuppressPrints():
            best_model.fit(dataset_train)
        #print("INFO: Final best model training complete.") # Log opcional seu
           
        # Predict on training and testing data
        y_pred_train = best_model.predict(dataset_train)
        y_pred_test = best_model.predict(dataset_test)
        #best_model.close_session()
        
        # Add the predictions and source to the training DataFrame
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
    
        # Add the predictions and source to the test DataFrame
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
    
        # Combine the training and testing DataFrames
        df_final = pd.concat([df_train, df_test], ignore_index=True)
    
        # Generate LIME explanations for first 10 test instances
        explanations = [
            explainer.explain_instance(
                X_test.iloc[i].values,
                lambda x_lime: best_model.predict_proba(pd.DataFrame(x_lime, columns=attributes)),
                num_features=X_test.shape[1]
            ) for i in range(min(10, len(X_test)))
        ]
        
        best_model.close_session()
    
        # Returns the combined DataFrame, model information, and LIME explanations
        return df_final, best_model, best_params, best_score, explanations
    
    else:
        print(
            "\nERRO no script principal: Não foi possível selecionar um trial com os critérios definidos. "
            "O treinamento otimizado não pode prosseguir."
        )
        return None, None, None, None, None



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Pos-processing Models                        """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def PostEOddsPostprocessingRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    Equalized Odds Postprocessing (EqOddsPostprocessing) to adjust the predictions on both 
    the training set and the test set, in order to reduce bias between privileged and 
    unprivileged groups. The Equalized Odds technique is a post-processing method that 
    adjusts the predictions of a model to reduce bias between groups. In other words, 
    it does not interfere with the training of the model itself, but adjusts the predictions 
    of the model after training. 
    
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
    
    # Create a list of feature names from the training DataFrame
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'
    
    # Prepare LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['0', '1'],
        discretize_continuous=True,
        random_state=SEED
    )
    
    # Step 1: Optimize RandomForest model using Optuna
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Number of estimators
        max_depth = trial.suggest_int('max_depth', 5, 20)  # Maximum depth
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)  # Minimum number of samples per sheet
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        # max_samples condicionalmente
        if bootstrap_enabled:
            # max_samples só é aplicável se bootstrap=True
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05) # step opcional para granularidade
        else:
            # Se bootstrap=False, max_samples deve ser None
            max_samples_val = None     

        # Instantiating the RandomForest model with hyperparameters 
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap_enabled,
            class_weight=class_weight,
            max_samples=max_samples_val,
            n_jobs=1,
            random_state=SEED
        )

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
            # Fit the model on the fold with instance weights
            clf_rf.fit(X_fold_train, y_fold_train)
            
            # Raw predictions
            y_val_pred = clf_rf.predict(X_fold_val)
                 
            # Prepare AIF360 datasets for postprocessing
            # Garantir nomes e índices consistentes para Series de labels
            y_fold_val_named = y_fold_val.copy()
            if not isinstance(y_fold_val_named, pd.Series) or y_fold_val_named.name is None:
                 y_fold_val_named = pd.Series(y_fold_val, name=label_col_name, index=X_fold_val.index)
            elif not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val.values, name=y_fold_val_named.name, index=X_fold_val.index)
            
            y_val_pred_series = pd.Series(y_val_pred, index=X_fold_val.index, name=label_col_name)
            
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val_named, protected_attribute)
            bd_pred = prepare_aif360_data(X_fold_val, y_val_pred_series, protected_attribute)

            #Como esse modelo usa TRP / FPR, classes muito desbalanceadas impactam os folds          
            # --- Início da Lógica Condicional para EqOdds ---
            is_priv = (bd_true.protected_attributes.ravel() == 0)
            is_unpriv = (bd_true.protected_attributes.ravel() == 1)
            is_neg_label = (bd_true.labels.ravel() == 0)
            
            num_priv_neg_actual = np.sum(is_priv & is_neg_label)
            num_unpriv_neg_actual = np.sum(is_unpriv & is_neg_label)
            
            # print(f"--- Debug Fold Info (Trial {trial.number}) ---") # Descomente para debug
            # print(f"  Privilegiados Negativos Reais: {num_priv_neg_actual}")
            # print(f"  Não Privilegiados Negativos Reais: {num_unpriv_neg_actual}")
            
            MIN_NEGATIVES_PER_GROUP = 1
            
            # Inicializa as métricas do fold com valores de penalidade/padrão
            f1_fold = 0.0
            di_fold = 0.0 # Um DI de 0 resulta em fairness_score = 0 (ruim)

            if num_priv_neg_actual < MIN_NEGATIVES_PER_GROUP or \
               num_unpriv_neg_actual < MIN_NEGATIVES_PER_GROUP:
                #print(f"AVISO - Trial {trial.number}, Fold: Instâncias negativas insuficientes para EqOdds. "
                      #f"Priv.Neg={num_priv_neg_actual}, Unpriv.Neg={num_unpriv_neg_actual}. "
                      #f"Atribuindo F1={f1_fold}, DI={di_fold} para este fold.")
                # f1_fold e di_fold já estão com valores de penalidade
                pass 
            else:
                # Prossiga com eq.fit e eq.predict
                try:
                    eq = EqOddsPostprocessing(
                        unprivileged_groups=[{protected_attribute: 1}],
                        privileged_groups=[{protected_attribute: 0}],
                        seed=SEED
                    )
                    eq.fit(bd_true, bd_pred)
                    bd_pred_post = eq.predict(bd_pred)
                    y_post_val = bd_pred_post.labels.ravel()

                    # Calcular métricas com as predições pós-processadas
                    f1_fold = f1_score(y_fold_val, y_post_val, average='weighted')
                    
                    df_val_metric = X_fold_val.copy() # Ou pd.DataFrame(X_fold_val, columns=attributes)
                    df_val_metric['Target'] = y_fold_val.values
                    df_val_metric['prediction'] = y_post_val
                    
                    di_fold = Metrics.ModelDisparateImpact(df_val_metric, protected_attribute, 
                                                           privileged_group=0, unprivileged_group=1,
                                                           prediction='prediction')
                
                except ValueError as e:
                    if "c must not contain values inf, nan, or None" in str(e) or \
                       "Unable to solve optimization problem" in str(e):
                        #print(f"AVISO - Trial {trial.number}, Fold: Erro no linprog ({e}). Atribuindo F1=0, DI=0.")
                        # f1_fold e di_fold permanecem nos valores de penalidade (0.0)
                        pass
                    else:
                        #print(f"ERRO - Trial {trial.number}, Fold: ValueError inesperado durante EqOdds ({e}). Atribuindo F1=0, DI=0.")
                        # f1_fold e di_fold permanecem nos valores de penalidade (0.0)
                        # Considere adicionar: raise e # Para erros realmente inesperados
                        pass
                except ZeroDivisionError as e:
                    #print(f"AVISO - Trial {trial.number}, Fold: ZeroDivisionError durante EqOdds ({e}). Atribuindo F1=0, DI=0.")
                    # f1_fold e di_fold permanecem nos valores de penalidade (0.0)
                    pass 
            
            f1_scores.append(f1_fold)
            di_scores.append(di_fold)
            # --- Fim da Lógica Condicional para EqOdds ---
                   
        
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)

        # Convert disparate impact into a fairness score in [0,1]
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        # Salvar os valores brutos de avg_f1 e avg_di para seleção customizada posterior
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
    
        # Return both objectives to maximize
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
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---") 
    best_trial_found = select_best_hyperparameters_trial(study) # Usa a função auxiliar
    
    if best_trial_found:
        best_params = best_trial_found.params
        best_score = best_trial_found.user_attrs.get("avg_f1") # F1 do trial selecionado
        final_di = best_trial_found.user_attrs.get("avg_di") # Adicionar para log
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---") # Adicionar
        if best_score is not None: # best_score é o F1
            print(f"F1-score (do trial selecionado): {best_score:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {best_params}")            
  
        # Train final model on full data
        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        # Raw predictions
        y_train_raw = best_model.predict(X_train)
        y_test_raw  = best_model.predict(X_test)
    
        # Garantir nome e índice consistentes para Series
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        y_test_named = y_test.copy()
        if y_test_named.name is None: y_test_named.name = label_col_name
        y_train_raw_series = pd.Series(y_train_raw, index=y_train_named.index, name=label_col_name)
        y_test_raw_series = pd.Series(y_test_raw, index=y_test_named.index, name=label_col_name)
    
        # Prepare AIF360 datasets for full postprocessing
        bd_train_true = prepare_aif360_data(X_train, y_train_named, protected_attribute)
        bd_train_pred = prepare_aif360_data(X_train, y_train_raw_series, protected_attribute)
        bd_test_pred  = prepare_aif360_data(X_test,  y_test_raw_series, protected_attribute)
        
        # Final Equalized Odds postprocessing
        eq_odds = EqOddsPostprocessing(
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}],
            seed=SEED
        )
        
        try:
            print("INFO: Ajustando o EqOddsPostprocessing final no conjunto de treino completo...")
            eq_odds.fit(bd_train_true, bd_train_pred)
        
            print("INFO: Aplicando EqOddsPostprocessing final aos dados de treino e teste...")
            y_train_pred_post = eq_odds.predict(bd_train_pred).labels.ravel()
            y_test_pred_post  = eq_odds.predict(bd_test_pred).labels.ravel()
        
            postprocessing_successful = True
        
        except ValueError as e:
            if "c must not contain values inf, nan, or None" in str(e) or \
               "Unable to solve optimization problem" in str(e):
                print(f"AVISO FINAL: Erro no linprog ao ajustar EqOdds no dataset completo ({e}). "
                      "As predições pós-processadas não estarão disponíveis. Usando predições brutas.")
            else:
                print(f"ERRO FINAL: ValueError inesperado durante EqOdds final ({e}). "
                      "Usando predições brutas.")
            postprocessing_successful = False
        except ZeroDivisionError as e:
            print(f"AVISO FINAL: ZeroDivisionError durante EqOdds final ({e}). "
                  "Usando predições brutas.")
            postprocessing_successful = False
        
        if not postprocessing_successful:
            # Se o pós-processamento falhou, use as predições brutas
            # ou decida retornar None para os resultados que dependem do pós-processamento.
            # Para LIME, precisaremos de probabilidades. Se EqOdds falhou,
            # a função post_predict_proba não funcionará como esperado.
            # Opção 1: Usar predições brutas para o df_final e para LIME explicar o modelo base.
            print("INFO: Pós-processamento falhou. O DataFrame final e LIME usarão predições brutas do modelo base.")
            y_train_pred_final = y_train_raw
            y_test_pred_final = y_test_raw
        
            # Ajustar a função post_predict_proba para LIME para usar o modelo base diretamente
            def base_model_predict_proba(X_lime_input):
                return best_model.predict_proba(X_lime_input)
        
            lime_predict_fn = base_model_predict_proba
        
        else: # Pós-processamento bem-sucedido
            y_train_pred_final = y_train_pred_post
            y_test_pred_final = y_test_pred_post
        
            # LIME explicações sobre as probabilidades PÓS-PROCESSADAS (como você já tinha)
            def post_predict_proba(X_lime_input):
                X_lime_input_df = pd.DataFrame(X_lime_input, columns=attributes)
                probs = best_model.predict_proba(X_lime_input_df) # Passar DataFrame para consistência se best_model for sklearn
                labels = np.argmax(probs, axis=1)
                # Cria a Series de labels com o mesmo índice do DataFrame de features do LIME
                labels_series = pd.Series(labels, name='Target', index=X_lime_input_df.index) 
                bd = prepare_aif360_data(X_lime_input_df, 
                                         labels_series, 
                                         protected_attribute)
                # Aplica o pós-processador JÁ TREINADO (eq_odds)
                bd_post = eq_odds.predict(bd) 
                return np.vstack([1 - bd_post.scores.ravel(), bd_post.scores.ravel()]).T
        
            lime_predict_fn = post_predict_proba
        
        # Adicionar as predições (pós-processadas ou brutas) aos DataFrames originais
        df_train['source'] = 'train'
        df_train['prediction'] = y_train_pred_final # Usar y_train_pred_final
        df_test['source'] = 'test'
        df_test['prediction'] = y_test_pred_final   # Usar y_test_pred_final
        
        df_final = pd.concat([df_train, df_test], ignore_index=True)
        
        # Generate LIME explanations
        explanations = [
            explainer.explain_instance(
                X_test.iloc[i].values,
                lime_predict_fn, # Usa a função de predição apropriada
                num_features=X_test.shape[1]
            ) for i in range(min(10, len(X_test)))
        ]
        
        # Retorna best_params que são os parâmetros do RandomForest.
        # Se EqOdds falhou, o usuário saberá pelo df_final (predições brutas)
        # e pelos logs. best_score ainda é o do Optuna (baseado em EqOdds nos folds).
        return df_final, best_model, best_params, best_score, explanations
        
    else:
        # Tratar caso onde nenhum trial foi selecionado
        print("\nERRO: Não foi possível selecionar um trial com os critérios definidos...")
        return None, None, None, None, None
    


def PostCalibratedEOddsRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies Calibrated Equalized Odds Postprocessing
    (CalibratedEqOddsPostprocessing) to adjust predictions on both the training and test sets,
    in order to reduce bias between privileged and underprivileged groups.

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
    
    # Create a list of feature names from the training DataFrame
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'
    
    # Initialize LIME explainer for final explanations
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['0', '1'],
        discretize_continuous=True,
        random_state=SEED
    )
    
    # Step 1: Optimize RandomForest model using Optuna
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Number of estimators
        max_depth = trial.suggest_int('max_depth', 5, 20)  # Maximum depth
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)  # Minimum number of samples per sheet
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        # max_samples condicionalmente
        if bootstrap_enabled:
            # max_samples só é aplicável se bootstrap=True
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05) # step opcional para granularidade
        else:
            # Se bootstrap=False, max_samples deve ser None
            max_samples_val = None     

        # Instantiating the RandomForest model with hyperparameters 
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap_enabled,
            class_weight=class_weight,
            max_samples=max_samples_val,
            n_jobs=1,
            random_state=SEED
        )
        
        # Define the cost constraints to be tested
        cost_constraint = trial.suggest_categorical("cost_constraint", ["fnr", "fpr", "weighted"])
        
        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            # Split data for this fold
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
            # Fit the model on the fold with instance weights
            clf_rf.fit(X_fold_train, y_fold_train)
            # Raw predictions
            y_val_pred = clf_rf.predict(X_fold_val)
            
            # Prepare AIF360 datasets for postprocessing
            # Garantir nomes e índices consistentes para Series de labels
            y_fold_val_named = y_fold_val.copy()
            if not isinstance(y_fold_val_named, pd.Series) or y_fold_val_named.name is None:
                 y_fold_val_named = pd.Series(y_fold_val, name=label_col_name, index=X_fold_val.index)
            elif not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val.values, name=y_fold_val_named.name, index=X_fold_val.index)
            
            y_val_pred_series = pd.Series(y_val_pred, index=X_fold_val.index, name=label_col_name)
            
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val_named, protected_attribute)
            bd_pred = prepare_aif360_data(X_fold_val, y_val_pred_series, protected_attribute)


            # --- Início da Lógica Condicional para CalibratedEqOdds ---
            is_priv = (bd_true.protected_attributes.ravel() == 0)
            is_unpriv = (bd_true.protected_attributes.ravel() == 1)
            is_neg_label = (bd_true.labels.ravel() == 0) 
            
            num_priv_neg_actual = np.sum(is_priv & is_neg_label)
            num_unpriv_neg_actual = np.sum(is_unpriv & is_neg_label)
            
            # Descomente para debug se necessário
            # print(f"--- Debug Fold Info (Trial {trial.number}, Cost Constraint: {cost_constraint}) ---")
            # print(f"  Privilegiados Negativos Reais: {num_priv_neg_actual}")
            # print(f"  Não Privilegiados Negativos Reais: {num_unpriv_neg_actual}")
            
            MIN_NEGATIVES_PER_GROUP = 1 # Ou um valor um pouco maior
            
            f1_fold = 0.0 
            di_fold = 0.0   

            if num_priv_neg_actual < MIN_NEGATIVES_PER_GROUP or \
               num_unpriv_neg_actual < MIN_NEGATIVES_PER_GROUP:
                #print(f"AVISO - Trial {trial.number}, Fold (Cost: {cost_constraint}): Instâncias negativas insuficientes. "
                      #f"Priv.Neg={num_priv_neg_actual}, Unpriv.Neg={num_unpriv_neg_actual}. "
                      #f"Atribuindo F1={f1_fold}, DI={di_fold}.")
               pass
            else:
                try:
                    post = CalibratedEqOddsPostprocessing( # Mudança aqui para CalibratedEqOdds
                        unprivileged_groups=[{protected_attribute: 1}],
                        privileged_groups=[{protected_attribute: 0}],
                        cost_constraint=cost_constraint, # Parâmetro específico
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
                       "constraint violation" in str(e).lower(): # Calibrated pode ter outros erros de otimização
                        #print(f"AVISO - Trial {trial.number}, Fold (Cost: {cost_constraint}): Erro no solver ({e}). Atribuindo F1=0, DI=0.")
                        pass
                    else:
                        #print(f"ERRO - Trial {trial.number}, Fold (Cost: {cost_constraint}): ValueError inesperado ({e}). Atribuindo F1=0, DI=0.")
                        # raise e # Opcional, para erros realmente inesperados
                        pass
                except ZeroDivisionError as e:
                    #print(f"AVISO - Trial {trial.number}, Fold (Cost: {cost_constraint}): ZeroDivisionError ({e}). Atribuindo F1=0, DI=0.")
                    pass
            
            f1_scores.append(f1_fold)
            di_scores.append(di_fold)
            # --- Fim da Lógica Condicional ---
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0 # Evita erro se f1_scores estiver vazio
        avg_di = np.mean(di_scores) if di_scores else 0.0   # Evita erro se di_scores estiver vazio

        fairness_score = 1.0 - abs(avg_di - 1.0)

        # Salvar os valores brutos de avg_f1 e avg_di para seleção customizada posterior
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
    
        # Return both objectives to maximize
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
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---") 
    best_trial_found = select_best_hyperparameters_trial(study) # Usa a função auxiliar
    
    if best_trial_found:
        final_best_params_to_return = best_trial_found.params.copy()         
        # Extrair o cost_constraint e manter o restante para o RandomForest
        params_for_model_instantiation = best_trial_found.params 
        cost_constraint_best = params_for_model_instantiation.pop('cost_constraint')
        rf_params_best = params_for_model_instantiation
        
        best_score = best_trial_found.user_attrs.get("avg_f1") # F1 do trial selecionado
        final_di = best_trial_found.user_attrs.get("avg_di") # Adicionar para log
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---") # Adicionar
        if best_score is not None: # best_score é o F1
            print(f"F1-score (do trial selecionado): {best_score:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {final_best_params_to_return}")  
           
        # Train final model on full data
        best_model = RandomForestClassifier(**rf_params_best, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        # Raw predictions
        y_train_raw = best_model.predict(X_train)
        y_test_raw  = best_model.predict(X_test)
        
        # Garantir nome e índice consistentes para Series
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        y_test_named = y_test.copy()
        if y_test_named.name is None: y_test_named.name = label_col_name
        y_train_raw_series = pd.Series(y_train_raw, index=y_train_named.index, name=label_col_name)
        y_test_raw_series = pd.Series(y_test_raw, index=y_test_named.index, name=label_col_name)
    
        # Prepare AIF360 datasets for full postprocessing
        bd_train_true = prepare_aif360_data(X_train, y_train_named, protected_attribute)
        bd_train_pred = prepare_aif360_data(X_train, y_train_raw_series, protected_attribute)
        bd_test_pred  = prepare_aif360_data(X_test,  y_test_raw_series, protected_attribute)
    
        # Final Equalized Odds postprocessing
        final_post = CalibratedEqOddsPostprocessing(
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}],
            cost_constraint=cost_constraint_best,
            seed=SEED,
        )
        
        y_train_pred_final = y_train_raw # Padrão se o pós-processamento falhar
        y_test_pred_final = y_test_raw   # Padrão se o pós-processamento falhar
        lime_predict_fn = lambda X_lime_input_df: best_model.predict_proba(X_lime_input_df) # Padrão LIME

        try:
            # print("INFO: Ajustando o CalibratedEqOddsPostprocessing final no conjunto de treino completo...")
            final_post.fit(bd_train_true, bd_train_pred)
            
            # print("INFO: Aplicando CalibratedEqOddsPostprocessing final...")
            y_train_pred_post = final_post.predict(bd_train_pred).labels.ravel()
            y_test_pred_post  = final_post.predict(bd_test_pred).labels.ravel()
            
            y_train_pred_final = y_train_pred_post # Atualiza se bem-sucedido
            y_test_pred_final = y_test_pred_post   # Atualiza se bem-sucedido

            def post_processed_predict_proba(X_lime_input_np):
                X_lime_df = pd.DataFrame(X_lime_input_np, columns=attributes)
                probs_raw = best_model.predict_proba(X_lime_df)
                labels_raw = np.argmax(probs_raw, axis=1)
                labels_raw_series = pd.Series(labels_raw, index=X_lime_df.index, name=label_col_name)
                
                bd_lime = prepare_aif360_data(X_lime_df, labels_raw_series, protected_attribute)
                bd_lime_post = final_post.predict(bd_lime) # Usa final_post treinado
                return np.vstack([1 - bd_lime_post.scores.ravel(), bd_lime_post.scores.ravel()]).T
            
            lime_predict_fn = post_processed_predict_proba
            print("INFO: Pós-processamento final aplicado com sucesso.")

        except ValueError as e:
            if "c must not contain values inf, nan, or None" in str(e) or \
               "Unable to solve optimization problem" in str(e) or \
               "constraint violation" in str(e).lower():
                #print(f"AVISO FINAL: Erro no solver ao ajustar CalibratedEqOdds no dataset completo ({e}). Usando predições brutas.")
                pass
            else:
                #print(f"ERRO FINAL: ValueError inesperado durante CalibratedEqOdds final ({e}). Usando predições brutas.")
                pass
        except ZeroDivisionError as e:
            #print(f"AVISO FINAL: ZeroDivisionError durante CalibratedEqOdds final ({e}). Usando predições brutas.")
            pass
        
        df_train['source'] = 'train'
        df_train['prediction'] = y_train_pred_final
        df_test['source'] = 'test'
        df_test['prediction'] = y_test_pred_final
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        explanations = [
            explainer.explain_instance(
                X_test.iloc[i].values,
                lime_predict_fn,
                num_features=X_test.shape[1] 
            ) for i in range(min(10, len(X_test)))
        ]
        
        return df_final, best_model, final_best_params_to_return, best_score, explanations
    else:
        print("\nERRO: Não foi possível selecionar um trial com os critérios definidos...")
        return None, None, None, None, None




