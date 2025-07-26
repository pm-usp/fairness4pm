# Libraries for data manipulation
import pandas as pd
import numpy as np
import random
import math

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
    dataset_train = BinaryLabelDataset(df=pd.concat([X_train, y_train], axis=1),
                                       label_names=['Target'],
                                       protected_attribute_names=[protected_attribute],
                                       favorable_label=1,
                                       unfavorable_label=0)

    dataset_test = BinaryLabelDataset(df=pd.concat([X_test, y_test], axis=1),
                                      label_names=['Target'],
                                      protected_attribute_names=[protected_attribute],
                                      favorable_label=1,
                                      unfavorable_label=0)

    return dataset_train, dataset_test


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                          Fairness Metric                                 """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def def_fairness_score(di, lower=0.8, upper=1.25):
    if lower <= di <= upper:
        return 1.0
        # distância até a fronteira mais próxima
    if di < lower:
        dist = lower - di
    else:  # di > upper
        dist = di - upper
    # normaliza pela "janela" máxima (0.25)
    score = max(0.0, 1.0 - dist / (upper - 1.0))
    return score

#Penalização quadrática (“smooth hinge”): pequenos desvios produzem apenas uma queda quadrática — muito menos 
#severa perto da fronteira — e somente aproximando do limite máximo da janela é que o score se aproxima de zero.

def def_fairness_score_quad(di, lower=0.8, upper=1.25):
    if lower <= di <= upper:
        return 1.0
    # distância até fronteira mais próxima
    dist = (lower - di) if di < lower else (di - upper)
    # escolhe a janela correta (assimétrico se desejar)
    window = (1.0 - lower) if di < lower else (upper - 1.0)
    return max(0.0, 1.0 - (dist/window)**2)


# Uma curva exponencial penaliza “de leve” no início e depois desce rápido, sem jamais zerar bruscamente

def def_fairness_score_exp(di, lower=0.8, upper=1.25, k=1.0):
    if lower <= di <= upper:
        return 1.0
    dist = (lower - di) if di < lower else (di - upper)
    window = (1.0 - lower) if di < lower else (upper - 1.0)
    # ângulo de quão rápido cai: k > 0 maior => queda mais acentuada
    return math.exp(-k * (dist/window))



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                          Pre-processing Models                           """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def PreReweighingRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, alpha, num_trials=50):
    """
    This function applies the AIF360 Reweighing technique to the training dataset and then 
    optimizes the hyperparameters of a RandomForestClassifier model using Optuna, 
    maximizing the F1 and minimizing the nonsense impact. 
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - alpha (float): Weight to balance the F1 score and nonsense impact in the objective function (the higher the alpha, the higher the F1 weight)
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

    # Convert the transformed dataset back to DataFrame
    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name='Target')
    
    # Objective function for multi-objective Optuna optimization
    weights = dataset_transf_train.instance_weights

    # Optimization function for Optuna
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Number of estimators
        max_depth = trial.suggest_int('max_depth', 5, 20)  # Maximum depth
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)  # Minimum number of samples per sheet
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap = trial.suggest_categorical('bootstrap', [True])  # Force the use of bootstrap to improve generalization
        class_weight = trial.suggest_categorical('class_weight', ['balanced'])  # Adjustment for imbalanced data
        max_samples = trial.suggest_float('max_samples', 0.6, 1.0)  # Sample a fraction of the data for each tree
        
        # Instantiating the RandomForest model with the suggested hyperparameters
        clf_rf = RandomForestClassifier(
             n_estimators=n_estimators,
             max_depth=max_depth,
             min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf,
             max_features=max_features,
             bootstrap=bootstrap,
             class_weight=class_weight,
             max_samples=max_samples,
             n_jobs=-1,
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
            df_val['Target'] = y_fold_val
            df_val['prediction'] = y_pred_val
    
            di = Metrics.DisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
    
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
            
        fairness_score = def_fairness_score_exp(avg_di, lower=0.8, upper=1.25)
        #fairness_score = 1.0 - abs(avg_di - 1.0)
        alpha = 0.5
    
        # Return both objectives to maximize
        return alpha * avg_f1 + (1 - alpha) * fairness_score

    
    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # Extract best parameters and score
    best = study.best_trial
    best_params = best.params.copy()
    best_score  = best.value
     
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
        class_names=['Target'],
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
    return df_final, best_model, best_params, best_score, explanations


def PreDisparateImpactRemoverRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, alpha, num_trials=50):
    """
    Applies the nonsense impact removal (DIR) technique to the training data and trains a RandomForest model,
    optimizing its hyperparameters with Optuna and using cross-validation. The function also calculates the nonsense impact and returns LIME explanations for the model's predictions.
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - alpha (float): Weight to balance the F1 score and nonsense impact in the objective function (the higher the alpha, the higher the F1 weight)
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
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name='Target')

    # Define Optuna objective for multi-objective optimization
    def objective(trial):
        # Hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 50, 200)  # Number of estimators
        max_depth = trial.suggest_int('max_depth', 5, 20)  # Maximum depth
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)  # Minimum number of samples per sheet
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap = trial.suggest_categorical('bootstrap', [True])  # Force the use of bootstrap to improve generalization
        class_weight = trial.suggest_categorical('class_weight', ['balanced'])  # Adjustment for imbalanced data
        max_samples = trial.suggest_float('max_samples', 0.6, 1.0)  # Sample a fraction of the data for each tree
        
        # Instantiating the RandomForest model with the suggested hyperparameters
        clf_rf = RandomForestClassifier(
             n_estimators=n_estimators,
             max_depth=max_depth,
             min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf,
             max_features=max_features,
             bootstrap=bootstrap,
             class_weight=class_weight,
             max_samples=max_samples,
             n_jobs=-1,
             random_state=SEED
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
            df_val['Target'] = y_fold_val
            df_val['prediction'] = y_pred_val
    
            di = Metrics.DisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
    
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
            
        fairness_score = def_fairness_score_exp(avg_di, lower=0.8, upper=1.25)
        #fairness_score = 1.0 - abs(avg_di - 1.0)
        alpha = 0.5
    
        # Return both objectives to maximize
        return alpha * avg_f1 + (1 - alpha) * fairness_score

    
    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # Extract best parameters and score
    best = study.best_trial
    best_params = best.params.copy()
    best_score  = best.value
     
    # Train final model on full transformed training data
    best_model = RandomForestClassifier(**best_params, random_state=SEED)
    best_model.fit(X_train_transf, y_train_transf)

    # Predict on training and testing data
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
         class_names=['Target'],
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

    # Returns the combined DataFrame, model information, and LIME explanations
    return df_final, best_model, best_params, best_score, explanations


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                       In-processing Models                               """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Required for compatibility with TensorFlow 1.x and AIF360
tf.compat.v1.disable_v2_behavior()

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
        try:
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
        finally:
            # garante que, se algo der errado no fit, a sessão é fechada
            self.sess.close()
            self.sess = None
        
        self.columns = dataset.feature_names  # Define columns after fitting the model
        return self

    def predict(self, dataset):
        """
        Performs label (class) predictions based on the provided input dataset.
        """
        if isinstance(dataset, np.ndarray):
            dataset = prepare_aif360_data(pd.DataFrame(dataset, columns=self.columns), np.zeros(len(dataset)), self.protected_attribute_name)
        predictions = self.model.predict(dataset)
        return predictions.labels.ravel()

    def predict_proba(self, dataset):
        """
        Performs probability predictions for the given dataset.
        Returns the predicted probabilities for each class (e.g. [prob_class_0, prob_class_1]).
        """
        if isinstance(dataset, np.ndarray):
            dataset = prepare_aif360_data(pd.DataFrame(dataset, columns=self.columns), np.zeros(len(dataset)), self.protected_attribute_name)
        predictions = self.model.predict(dataset)
        proba = np.vstack([1 - predictions.scores.ravel(), predictions.scores.ravel()]).T
        # Check if the probabilities sum to 1
        print("Probas:", proba[:5], "Sums:", proba.sum(axis=1)[:5])  # Add print to check probabilities
        if not np.allclose(proba.sum(axis=1), 1):
            raise ValueError("Prediction probabilities do not sum to 1.")
        return proba

    def close_session(self):
        """
        Ends the TensorFlow session to free up memory resources.
        """
        if self.sess is not None:
            self.sess.close()

def prepare_aif360_data(X, y, protected_attribute_name):
    """
    Prepares input data in the format required by the AIF360 library for inclusion of protected attributes.
    The goal is to create a BinaryLabelDataset with the label, protected attributes, and privileged values ​​defined.
    """
    df = X.copy()
    df['Target'] = y
    return StandardDataset(df, label_name='Target', favorable_classes=[1],
                           protected_attribute_names=[protected_attribute_name], privileged_classes=[[0]])


def InAdversarialDebiasingOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, alpha, num_trials=50):
    """
    Implements the Adversarial Debiasing model with AIF360 and TensorFlow.
    This model fine-tunes predictions to remove bias towards protected attributes (such as gender or race).
    The model is optimized using the Optuna framework to perform hyperparameter optimizations, and the final predictions are returned, along with LIME explanations for samples in the test set.
    
    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - alpha (float): Weight to balance the F1 score and nonsense impact in the objective function (the higher the alpha, the higher the F1 weight)
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
            dataset_val = prepare_aif360_data(X_fold_val, y_fold_val, protected_attribute)
            
            #Treina e avalia
            model.fit(dataset_train)
            
            # predição no fold de validação
            y_pred_val = model.predict(dataset_val)
            model.close_session()

            # Compute F1 score
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            # Compute disparate impact
            df_val = pd.DataFrame(X_fold_val, columns=attributes)
            df_val['Target'] = y_fold_val
            df_val['prediction'] = y_pred_val
            
            di = Metrics.DisparateImpact(df_val, protected_attribute, 
                                         privileged_group=0, unprivileged_group=1,
                                         prediction='prediction')
            di_scores.append(di)
                   
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
            
        fairness_score = def_fairness_score_exp(avg_di, lower=0.8, upper=1.25)
        #fairness_score = 1.0 - abs(avg_di - 1.0)
        alpha = 0.5
    
        # Return both objectives to maximize
        return alpha * avg_f1 + (1 - alpha) * fairness_score

    
    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # Extract best parameters and score
    best = study.best_trial
    best_params = best.params.copy()
    best_score  = best.value     
    
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
    best_model.fit(dataset_train)

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

    # Compute LIME explanations for the first 10 samples of the test set
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Target'], discretize_continuous=True)
    
    def predict_proba_fn(data):
        """
        Auxiliary function to obtain the prediction probabilities of the adversarial model.
        """
        return best_model.predict_proba(data).astype(np.float64)
    
    explanations = [explainer.explain_instance(X_test.iloc[i].values, 
                                               predict_proba_fn, num_features=X_test.shape[1]) 
                                               for i in range(min(10, len(X_test)))]
    best_model.close_session()

    # Returns the combined DataFrame, model information, and LIME explanations
    return df_final, best_model, best_params, best_score, explanations


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Pos-processing Models                        """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def PostEOddsPostprocessingRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, alpha, num_trials=50):
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
    - alpha (float): Weight to balance the F1 score and nonsense impact in the objective function (the higher the alpha, the higher the F1 weight)
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
        bootstrap = trial.suggest_categorical('bootstrap', [True])  # Force the use of bootstrap to improve generalization
        class_weight = trial.suggest_categorical('class_weight', ['balanced'])  # Adjustment for imbalanced data
        max_samples = trial.suggest_float('max_samples', 0.6, 1.0)  # Sample a fraction of the data for each tree        

        # Instantiating the RandomForest model with hyperparameters 
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            max_samples=max_samples,
            n_jobs=-1,
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
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val, protected_attribute)
            bd_pred = prepare_aif360_data(X_fold_val, pd.Series(y_val_pred, index=y_fold_val.index, name='Target'), protected_attribute)

            # Apply Equalized Odds postprocessing
            eq = EqOddsPostprocessing(
                unprivileged_groups=[{protected_attribute: 1}],
                privileged_groups=[{protected_attribute: 0}],
                seed=SEED
            )
            eq.fit(bd_true, bd_pred)
            bd_pred_post = eq.predict(bd_pred)
            y_post_val = bd_pred_post.labels.ravel()

            # Compute F1 score
            f1 = f1_score(y_fold_val, y_post_val, average='weighted')
            f1_scores.append(f1)
            
            # Compute disparate impact
            df_val = pd.DataFrame(X_fold_val, columns=attributes)
            df_val['Target'] = y_fold_val
            df_val['prediction'] = y_post_val
           
            di = Metrics.DisparateImpact(df_val, protected_attribute, 
                                         privileged_group=0, unprivileged_group=1,
                                         prediction='prediction')
            di_scores.append(di)
                   
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)
           
        fairness_score = def_fairness_score_exp(avg_di, lower=0.8, upper=1.25)
        #fairness_score = 1.0 - abs(avg_di - 1.0)
        alpha = 0.5
    
        # Return both objectives to maximize
        return alpha * avg_f1 + (1 - alpha) * fairness_score

    
    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # Extract best parameters and score
    best = study.best_trial
    best_params = best.params.copy()
    best_score  = best.value

    # Train final model on full data
    best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    # Raw predictions
    y_train_raw = best_model.predict(X_train)
    y_test_raw  = best_model.predict(X_test)

    # Prepare AIF360 datasets for full postprocessing
    bd_train_true = prepare_aif360_data(X_train, y_train, protected_attribute)
    bd_train_pred = prepare_aif360_data(X_train, pd.Series(y_train_raw, index=y_train.index, name='Target'), protected_attribute)
    bd_test_pred  = prepare_aif360_data(X_test,  pd.Series(y_test_raw,  index=y_test.index, name='Target'), protected_attribute)

    # Final Equalized Odds postprocessing
    eq_odds = EqOddsPostprocessing(
        unprivileged_groups=[{protected_attribute: 1}],
        privileged_groups=[{protected_attribute: 0}],
        seed=SEED
    )
    
    # Fit EqOdds to the training set
    eq_odds.fit(bd_train_true, bd_train_pred)
    
    # Apply EqOdds to training and testing predictions
        # Extract the post-processed predictions
    y_train_pred_post = eq_odds.predict(bd_train_pred).labels.ravel()
    y_test_pred_post  = eq_odds.predict(bd_test_pred).labels.ravel()        

    # Add the adjusted predictions to the original training and testing DataFrames
    df_train['source'] = 'train'
    df_train['prediction'] = y_train_pred_post
    df_test['source'] = 'test'
    df_test['prediction'] = y_test_pred_post    

    # Combine the training and testing DataFrames
    df_final = pd.concat([df_train, df_test], ignore_index=True)

    # Generate LIME explanations for the first 10 test instances    
    # LIME explanations on post-processed probabilities
    def post_predict_proba(X):
        # Se vier um array ou lista, converta em DataFrame com as colunas de X_train
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=X_train.columns)
        # Raw probabilities
        probs = best_model.predict_proba(X)
        # Predict labels and build AIF360 dataset
        labels = np.argmax(probs, axis=1)
        bd = prepare_aif360_data(pd.DataFrame(X, columns=X_train.columns), pd.Series(labels, name='Target'), protected_attribute)
        bd_post = eq_odds.predict(bd)
        # Return adjusted probability of class 1
        return np.vstack([1 - bd_post.scores.ravel(), bd_post.scores.ravel()]).T

    explanations = [
        explainer.explain_instance(
            X_test.iloc[i].values,
            post_predict_proba,
            num_features=X_test.shape[1]
        ) for i in range(min(10, len(X_test)))
    ]

    # Returns the combined DataFrame, model information, and LIME explanations
    return df_final, best_model, best_params, best_score, explanations

    

def PostCalibratedEOddsRandomFlorestOptuna(X_train, y_train, X_test, y_test, df_train, df_test, protected_attribute, alpha, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies Calibrated Equalized Odds Postprocessing
    (CalibratedEqOddsPostprocessing) to adjust predictions on both the training and test sets,
    in order to reduce bias between privileged and underprivileged groups.

    Parameters:
    - X_train, y_train (pd.DataFrame, pd.Series): Training set with features and target.
    - X_test, y_test (pd.DataFrame, pd.Series): Test set with features and target.
    - df_train, df_test (pd.DataFrame): Original training and test sets, where the predictions will be added.
    - protected_attribute (str): Name of the protected attribute for the fairness technique.
    - alpha (float): Weight to balance the F1 score and nonsense impact in the objective function (the higher the alpha, the higher the F1 weight)
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
        bootstrap = trial.suggest_categorical('bootstrap', [True])  # Force the use of bootstrap to improve generalization
        class_weight = trial.suggest_categorical('class_weight', ['balanced'])  # Adjustment for imbalanced data
        max_samples = trial.suggest_float('max_samples', 0.6, 1.0)  # Sample a fraction of the data for each tree

        
        # Instantiating the RandomForest model with hyperparameters 
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            max_samples=max_samples,
            n_jobs=-1,
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
            bd_true = prepare_aif360_data(X_fold_val, y_fold_val, protected_attribute)
            bd_pred = prepare_aif360_data(X_fold_val, pd.Series(y_val_pred, index=y_fold_val.index, name='Target'), protected_attribute)

            # Apply Calibrated Equalized Odds
            post = CalibratedEqOddsPostprocessing(
                unprivileged_groups=[{protected_attribute: 1}],
                privileged_groups=[{protected_attribute: 0}],
                cost_constraint=cost_constraint,
                seed=SEED
            )
            post.fit(bd_true, bd_pred)
            bd_pred_post = post.predict(bd_pred)
            y_post_val = bd_pred_post.labels.ravel()

            # Compute F1 score
            f1 = f1_score(y_fold_val, y_post_val, average='weighted')
            f1_scores.append(f1)
            
            # Compute disparate impact
            df_val = pd.DataFrame(X_fold_val, columns=attributes)
            df_val['Target'] = y_fold_val
            df_val['prediction'] = y_post_val
           
            di = Metrics.DisparateImpact(df_val, protected_attribute, 
                                         privileged_group=0, unprivileged_group=1,
                                         prediction='prediction')
            di_scores.append(di)
                   
        # Calculate average scores across folds
        avg_f1 = np.mean(f1_scores)
        avg_di = np.mean(di_scores)

        fairness_score = def_fairness_score_exp(avg_di, lower=0.8, upper=1.25)
        #fairness_score = 1.0 - abs(avg_di - 1.0)
        alpha = 0.5
    
        # Return both objectives to maximize
        return alpha * avg_f1 + (1 - alpha) * fairness_score

    
    # Configure Optuna study for multi-objective optimization with pruning
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # Extract best parameters and score
    best = study.best_trial
    best_params = best.params.copy()
    best_score  = best.value
    
    # Remove 'cost_constraint' from best hyperparameters as it is not part of RandomForestClassifier
    rf_params = {k: v for k, v in best_params.items() if k != 'cost_constraint'}
    best_cost = best_params.pop("cost_constraint")

    # Train final model on full data
    best_model = RandomForestClassifier(**rf_params, random_state=SEED, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    # Raw predictions
    y_train_raw = best_model.predict(X_train)
    y_test_raw  = best_model.predict(X_test)

    # Prepare AIF360 datasets for full postprocessing
    bd_train_true = prepare_aif360_data(X_train, y_train, protected_attribute)
    bd_train_pred = prepare_aif360_data(X_train, pd.Series(y_train_raw, index=y_train.index, name='Target'), protected_attribute)
    bd_test_pred  = prepare_aif360_data(X_test,  pd.Series(y_test_raw,  index=y_test.index, name='Target'), protected_attribute)

    # Final Equalized Odds postprocessing
    final_post = CalibratedEqOddsPostprocessing(
        unprivileged_groups=[{protected_attribute: 1}],
        privileged_groups=[{protected_attribute: 0}],
        cost_constraint=best_cost,
        seed=SEED
    )
    
    # Fit EqOdds to the training set
    final_post.fit(bd_train_true, bd_train_pred)
    
    # Apply EqOdds to training and testing predictions
        # Extract the post-processed predictions
    y_train_pred_post = final_post.predict(bd_train_pred).labels.ravel()
    y_test_pred_post  = final_post.predict(bd_test_pred).labels.ravel()
    
    # Add the adjusted predictions to the original training and testing DataFrames
    df_train['source'] = 'train'
    df_train['prediction'] = y_train_pred_post
    df_test['source'] = 'test'
    df_test['prediction'] = y_test_pred_post

    # Combine the training and testing DataFrames
    df_final = pd.concat([df_train, df_test], ignore_index=True)
    
    # Generate LIME explanations for the first 10 test instances    
    # LIME explanations on post-processed probabilities
    def post_predict_proba(X):
        # Se vier um array ou lista, converta em DataFrame com as colunas de X_train
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=X_train.columns)
        # Raw probabilities
        probs = best_model.predict_proba(X)
        # Predict labels and build AIF360 dataset
        labels = np.argmax(probs, axis=1)
        bd = prepare_aif360_data(pd.DataFrame(X, columns=X_train.columns), pd.Series(labels, name='Target'), protected_attribute)
        bd_post = final_post.predict(bd)
        # Return adjusted probability of class 1
        return np.vstack([1 - bd_post.scores.ravel(), bd_post.scores.ravel()]).T

    explanations = [
        explainer.explain_instance(
            X_test.iloc[i].values,
            post_predict_proba,
            num_features=X_test.shape[1]
        ) for i in range(min(10, len(X_test)))
    ]

   # Returns the combined DataFrame, model information, and LIME explanations
    return df_final, best_model, best_params, best_score, explanations




