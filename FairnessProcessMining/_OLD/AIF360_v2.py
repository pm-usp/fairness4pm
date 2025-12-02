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
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target' # Definir nome do label

    # Create datasets for the AIF360 library
    # Supondo que AIF360Datasets foi corrigida para não usar 'feature_names' diretamente em BinaryLabelDataset se der erro
    dataset_train, dataset_test = AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute)

    RW = Reweighing(unprivileged_groups=[{protected_attribute: 1}],
                    privileged_groups=[{protected_attribute: 0}])
    dataset_transf_train = RW.fit_transform(dataset_train) 
    
    # Assegurar que 'attributes' reflita as features realmente usadas após AIF360, se houver mudança
    # Se AIF360Datasets e Reweighing não mudam as feature_names de X_train, 'attributes' continua válido.
    # Caso contrário, usar: current_model_features = dataset_transf_train.feature_names
    current_model_features = list(dataset_transf_train.feature_names)
    if attributes != current_model_features:
        print(f"Aviso: Lista de features original ('attributes') mudou após transformações AIF360. Usando: {current_model_features}")
        attributes = current_model_features # Atualiza attributes para o que o modelo realmente verá

    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    # Assegurar consistência de índice
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name=label_col_name, index=X_train_transf.index)
    weights = pd.Series(dataset_transf_train.instance_weights, index=X_train_transf.index)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
    
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED,
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
        
        for train_index, val_index in skf.split(X_train_transf, y_train_transf):
            X_fold_train, X_fold_val = X_train_transf.iloc[train_index], X_train_transf.iloc[val_index]
            y_fold_train, y_fold_val = y_train_transf.iloc[train_index], y_train_transf.iloc[val_index]
            weights_fold_train = weights.iloc[train_index] # Usar .iloc se weights for Series com índice não padrão
            
            clf_rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
            
            y_pred_val = clf_rf.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            df_val = X_fold_val.copy() # Mais seguro que recriar com pd.DataFrame e columns=attributes
            df_val[label_col_name] = y_fold_val.values # Usar label_col_name
            df_val['prediction'] = y_pred_val
            
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
                
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- SEÇÃO DE VISUALIZAÇÃO DO OPTUNA ---
    print("\n--- Gerando Visualizações do Estudo Optuna ---")
    if len(study.trials) > 0: # Verifica se há trials para plotar
        try:
            # Gráfico da Fronteira de Pareto
            # Os nomes dos objetivos são inferidos da ordem que você retorna em 'objective'
            # ou você pode especificá-los com target_names
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio", "Fairness Score Médio"])
            fig_pareto.show() 
            # Para salvar: fig_pareto.write_html("PreReweighing_pareto_front.html")

            # Histórico da Otimização
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # Para salvar: fig_history.write_html("PreReweighing_optimization_history.html")

            # Slice Plot para ver o impacto de cada hiperparâmetro
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # Para salvar: fig_slice.write_html("PreReweighing_slice_plot.html")

            # Importância dos Parâmetros para o F1-score
            fig_importance_f1 = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[0] if t.values is not None else float('nan'), # Primeiro valor retornado pela 'objective' (avg_f1)
                target_name="Importância para F1-score"
            )
            fig_importance_f1.show()
            # Para salvar: fig_importance_f1.write_html("PreReweighing_param_importances_f1.html")

            # Importância dos Parâmetros para o Fairness Score
            fig_importance_fs = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[1] if t.values is not None else float('nan'), # Segundo valor retornado pela 'objective' (fairness_score)
                target_name="Importância para Fairness Score"
            )
            fig_importance_fs.show()
            # Para salvar: fig_importance_fs.write_html("PreReweighing_param_importances_fs.html")

        except Exception as e:
            print(f"Erro ao gerar visualizações do Optuna: {e}")
            print("Verifique se Plotly está instalado e se há trials suficientes e válidos no estudo.")
    else:
        print("Nenhum trial no estudo para gerar visualizações.")
    # --- FIM DA SEÇÃO DE VISUALIZAÇÃO ---
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params
        final_f1 = best_trial_found.user_attrs.get("avg_f1") # Nomeado como final_f1 aqui
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---")
        if final_f1 is not None:
            print(f"F1-score (CV do trial selecionado): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {best_params}")

        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1) # n_jobs=1 já estava
        best_model.fit(X_train_transf, y_train_transf, sample_weight=weights)
        
        y_pred_train = best_model.predict(X_train_transf)
        y_pred_test = best_model.predict(X_test) # X_test original, não transformado por Reweighing
        
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
        
        df_final = pd.concat([df_train, df_test], ignore_index=True)
        
        # Initialize LIME explainer for final explanations (já inicializado no topo)
        # explainer = LimeTabularExplainer(...)
        
        explanations = []
        # Para LIME, usar as features originais de X_test (não as transformadas por Reweighing,
        # pois o modelo final foi treinado em X_train_transf, mas LIME deve explicar
        # em termos das features que o usuário entende, e o predict_proba precisa de dados
        # no formato que o modelo espera)
        # A lambda function precisa que os dados sejam formatados como X_train_transf
        
        # X_test_for_lime DEVE ter as mesmas colunas que 'attributes' (que são as colunas de X_train_transf)
        # Se X_test original não tem as mesmas colunas que X_train (e.g. após dummização apenas em X_train),
        # precisa de um pré-processamento consistente.
        # Assumindo que X_test já está no formato esperado pelas colunas de 'attributes'
        # ou que você tem um X_test_processed que corresponde a 'attributes'.
        # Se 'attributes' são as colunas de X_train_transf, então a lambda precisa de dados nesse formato.
        
        # O explainer foi treinado com X_train.values e feature_names=X_train.columns.tolist() (originais)
        # O best_model foi treinado com X_train_transf
        # Isso é uma INCONSISTÊNCIA para LIME. LIME deve ser treinado/configurado
        # com os dados no mesmo formato que o modelo a ser explicado espera.

        # CORREÇÃO PARA LIME:
        # O explainer deve ser configurado com X_train_transf
        # Ou, se quiser explicar em features originais, a lambda predict_fn do LIME precisa
        # fazer a mesma transformação de X_train para X_train_transf
        # Abordagem mais simples: LIME explica o modelo que age em X_train_transf

        lime_explainer_for_transformed = LimeTabularExplainer(
            training_data=X_train_transf.values, # Explicar modelo que opera em dados transformados
            feature_names=attributes, # Nomes das colunas de X_train_transf
            class_names=[label_col_name if label_col_name else '0', '1'], # ['0', '1'] ou nomes reais
            discretize_continuous=True,
            random_state=SEED
        )

        for i in range(min(10, X_test.shape[0])):
            # X_test precisa ser transformado da mesma forma que X_train foi para X_train_transf
            # AIF360 Reweighing não transforma X_test. O modelo é treinado em X_train_transf.
            # Para predição, X_test é usado diretamente, o que é correto para avaliar o modelo treinado
            # em dados não vistos NO FORMATO ORIGINAL (se o modelo não espera transformação em X_test).
            # MAS, o best_model foi treinado em X_train_transf.
            # Portanto, para LIME explicar esse best_model, ele precisa de dados no formato de X_train_transf.
            # E a função predict_proba do LIME deve receber dados nesse formato.

            # Se X_test não passou pelo mesmo pré-processamento que X_train para virar X_train_transf
            # (Reweighing é só no treino para os pesos, as features de X_test não mudam por Reweighing),
            # então X_test já está no formato "original" de features.
            # O 'best_model' foi treinado em 'X_train_transf'.
            # A predição 'y_pred_test = best_model.predict(X_test)' implica que
            # as colunas de X_test DEVEM ser as mesmas de X_train_transf (que são 'attributes').
            # Isso está correto se 'attributes' são as features originais e Reweighing não muda as features.
            # (Reweighing muda pesos, não features em si).
            
            # Dado que `attributes = X_train.columns.tolist()` e `X_train_transf` usa essas mesmas colunas
            # (Reweighing não altera as colunas de features, apenas os pesos de instância),
            # então X_test deve ter as mesmas colunas `attributes`.

            instance_to_explain = X_test.iloc[i].values 
            
            exp = lime_explainer_for_transformed.explain_instance(
                data_row=instance_to_explain, # Deve ter o mesmo número de features que X_train_transf
                predict_fn=lambda x_lime: best_model.predict_proba(pd.DataFrame(x_lime, columns=attributes)),
                num_features=len(attributes) 
            )
            explanations.append(exp)
        
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
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target' # Definir nome do label

    # Create datasets for the AIF360 library
    # Supondo que AIF360Datasets foi corrigida para não usar 'feature_names' diretamente em BinaryLabelDataset se der erro
    dataset_train, dataset_test = AIF360Datasets(X_train, y_train, X_test, y_test, protected_attribute)

    DIR = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=protected_attribute)  # Complete repair
    dataset_transf_train = DIR.fit_transform(dataset_train) 
    
    # Assegurar que 'attributes' reflita as features realmente usadas após AIF360, se houver mudança
    # Se AIF360Datasets e Reweighing não mudam as feature_names de X_train, 'attributes' continua válido.
    # Caso contrário, usar: current_model_features = dataset_transf_train.feature_names
    current_model_features = list(dataset_transf_train.feature_names)
    if attributes != current_model_features:
        print(f"Aviso: Lista de features original ('attributes') mudou após transformações AIF360. Usando: {current_model_features}")
        attributes = current_model_features # Atualiza attributes para o que o modelo realmente verá

    X_train_transf = pd.DataFrame(dataset_transf_train.features, columns=attributes)
    # Assegurar consistência de índice
    y_train_transf = pd.Series(dataset_transf_train.labels.ravel(), name=label_col_name, index=X_train_transf.index)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
    
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED,
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
        
        for train_index, val_index in skf.split(X_train_transf, y_train_transf):
            X_fold_train, X_fold_val = X_train_transf.iloc[train_index], X_train_transf.iloc[val_index]
            y_fold_train, y_fold_val = y_train_transf.iloc[train_index], y_train_transf.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            
            y_pred_val = clf_rf.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred_val, average='weighted')
            f1_scores.append(f1)
            
            df_val = X_fold_val.copy() # Mais seguro que recriar com pd.DataFrame e columns=attributes
            df_val[label_col_name] = y_fold_val.values # Usar label_col_name
            df_val['prediction'] = y_pred_val
            
            di = Metrics.ModelDisparateImpact(df_val, protected_attribute, privileged_group=0, unprivileged_group=1, prediction='prediction')
            di_scores.append(di)
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
                
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- SEÇÃO DE VISUALIZAÇÃO DO OPTUNA ---
    print("\n--- Gerando Visualizações do Estudo Optuna ---")
    if len(study.trials) > 0: # Verifica se há trials para plotar
        try:
            # Gráfico da Fronteira de Pareto
            # Os nomes dos objetivos são inferidos da ordem que você retorna em 'objective'
            # ou você pode especificá-los com target_names
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio", "Fairness Score Médio"])
            fig_pareto.show() 
            # Para salvar: fig_pareto.write_html("PreReweighing_pareto_front.html")

            # Histórico da Otimização
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # Para salvar: fig_history.write_html("PreReweighing_optimization_history.html")

            # Slice Plot para ver o impacto de cada hiperparâmetro
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # Para salvar: fig_slice.write_html("PreReweighing_slice_plot.html")

            # Importância dos Parâmetros para o F1-score
            fig_importance_f1 = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[0] if t.values is not None else float('nan'), # Primeiro valor retornado pela 'objective' (avg_f1)
                target_name="Importância para F1-score"
            )
            fig_importance_f1.show()
            # Para salvar: fig_importance_f1.write_html("PreReweighing_param_importances_f1.html")

            # Importância dos Parâmetros para o Fairness Score
            fig_importance_fs = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[1] if t.values is not None else float('nan'), # Segundo valor retornado pela 'objective' (fairness_score)
                target_name="Importância para Fairness Score"
            )
            fig_importance_fs.show()
            # Para salvar: fig_importance_fs.write_html("PreReweighing_param_importances_fs.html")

        except Exception as e:
            print(f"Erro ao gerar visualizações do Optuna: {e}")
            print("Verifique se Plotly está instalado e se há trials suficientes e válidos no estudo.")
    else:
        print("Nenhum trial no estudo para gerar visualizações.")
    # --- FIM DA SEÇÃO DE VISUALIZAÇÃO ---
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params
        final_f1 = best_trial_found.user_attrs.get("avg_f1") # Nomeado como final_f1 aqui
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---")
        if final_f1 is not None:
            print(f"F1-score (CV do trial selecionado): {final_f1:.4f}")
        if final_di is not None:
            print(f"Disparate Impact (DI) (CV do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros: {best_params}")

        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1) # n_jobs=1 já estava
        best_model.fit(X_train_transf, y_train_transf)
        
        y_pred_train = best_model.predict(X_train_transf)
        y_pred_test = best_model.predict(X_test) # X_test original, não transformado 
        
        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
        
        df_final = pd.concat([df_train, df_test], ignore_index=True)
        
        # Initialize LIME explainer for final explanations (já inicializado no topo)
        # explainer = LimeTabularExplainer(...)
        
        explanations = []
        # Para LIME, usar as features originais de X_test (não as transformadas por Reweighing,
        # pois o modelo final foi treinado em X_train_transf, mas LIME deve explicar
        # em termos das features que o usuário entende, e o predict_proba precisa de dados
        # no formato que o modelo espera)
        # A lambda function precisa que os dados sejam formatados como X_train_transf
        
        # X_test_for_lime DEVE ter as mesmas colunas que 'attributes' (que são as colunas de X_train_transf)
        # Se X_test original não tem as mesmas colunas que X_train (e.g. após dummização apenas em X_train),
        # precisa de um pré-processamento consistente.
        # Assumindo que X_test já está no formato esperado pelas colunas de 'attributes'
        # ou que você tem um X_test_processed que corresponde a 'attributes'.
        # Se 'attributes' são as colunas de X_train_transf, então a lambda precisa de dados nesse formato.
        
        # O explainer foi treinado com X_train.values e feature_names=X_train.columns.tolist() (originais)
        # O best_model foi treinado com X_train_transf
        # Isso é uma INCONSISTÊNCIA para LIME. LIME deve ser treinado/configurado
        # com os dados no mesmo formato que o modelo a ser explicado espera.

        # CORREÇÃO PARA LIME:
        # O explainer deve ser configurado com X_train_transf
        # Ou, se quiser explicar em features originais, a lambda predict_fn do LIME precisa
        # fazer a mesma transformação de X_train para X_train_transf
        # Abordagem mais simples: LIME explica o modelo que age em X_train_transf

        lime_explainer_for_transformed = LimeTabularExplainer(
            training_data=X_train_transf.values, # Explicar modelo que opera em dados transformados
            feature_names=attributes, # Nomes das colunas de X_train_transf
            class_names=[label_col_name if label_col_name else '0', '1'], # ['0', '1'] ou nomes reais
            discretize_continuous=True,
            random_state=SEED
        )

        for i in range(min(10, X_test.shape[0])):
            # X_test precisa ser transformado da mesma forma que X_train foi para X_train_transf
            # AIF360 Reweighing não transforma X_test. O modelo é treinado em X_train_transf.
            # Para predição, X_test é usado diretamente, o que é correto para avaliar o modelo treinado
            # em dados não vistos NO FORMATO ORIGINAL (se o modelo não espera transformação em X_test).
            # MAS, o best_model foi treinado em X_train_transf.
            # Portanto, para LIME explicar esse best_model, ele precisa de dados no formato de X_train_transf.
            # E a função predict_proba do LIME deve receber dados nesse formato.

            # Se X_test não passou pelo mesmo pré-processamento que X_train para virar X_train_transf
            # (Reweighing é só no treino para os pesos, as features de X_test não mudam por Reweighing),
            # então X_test já está no formato "original" de features.
            # O 'best_model' foi treinado em 'X_train_transf'.
            # A predição 'y_pred_test = best_model.predict(X_test)' implica que
            # as colunas de X_test DEVEM ser as mesmas de X_train_transf (que são 'attributes').
            # Isso está correto se 'attributes' são as features originais e Reweighing não muda as features.
            # (Reweighing muda pesos, não features em si).
            
            # Dado que `attributes = X_train.columns.tolist()` e `X_train_transf` usa essas mesmas colunas
            # (Reweighing não altera as colunas de features, apenas os pesos de instância),
            # então X_test deve ter as mesmas colunas `attributes`.

            instance_to_explain = X_test.iloc[i].values 
            
            exp = lime_explainer_for_transformed.explain_instance(
                data_row=instance_to_explain, # Deve ter o mesmo número de features que X_train_transf
                predict_fn=lambda x_lime: best_model.predict_proba(pd.DataFrame(x_lime, columns=attributes)),
                num_features=len(attributes) 
            )
            explanations.append(exp)
        
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


def InAdversarialDebiasingOptuna(X_train, y_train, X_test, y_test, 
                                 df_train, df_test, 
                                 protected_attribute, num_trials=50):
    """
    Implements the Adversarial Debiasing model with AIF360 and TensorFlow
    to mitigate bias towards protected attributes during model training.
    The model's hyperparameters are optimized using Optuna, balancing predictive
    performance (F1-score) and fairness (Disparate Impact close to 1).
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=attributes, # Usa 'attributes' que são X_train.columns.tolist()
        class_names=['0', '1'], 
        discretize_continuous=True,
        random_state=SEED
    )

    def objective(trial):
        num_epochs = trial.suggest_categorical('num_epochs', [50, 100, 200])
        units = trial.suggest_categorical('classifier_num_hidden_units', [50, 100, 200])
        weight = trial.suggest_float('adversary_loss_weight', 0.1, 2.0, step=0.1) # step opcional

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            # Garantir nomes e índices para y_fold_train e y_fold_val
            y_fold_train_named = y_fold_train.copy()
            if y_fold_train_named.name is None: y_fold_train_named.name = label_col_name
            if not y_fold_train_named.index.equals(X_fold_train.index):
                 y_fold_train_named = pd.Series(y_fold_train_named.values, index=X_fold_train.index, name=y_fold_train_named.name)

            y_fold_val_named = y_fold_val.copy()
            if y_fold_val_named.name is None: y_fold_val_named.name = label_col_name
            if not y_fold_val_named.index.equals(X_fold_val.index):
                 y_fold_val_named = pd.Series(y_fold_val_named.values, index=X_fold_val.index, name=y_fold_val_named.name)

            model = AdversarialDebiasingWrapper(
                protected_attribute_name=protected_attribute,
                privileged_groups=[{protected_attribute: 0}],
                unprivileged_groups=[{protected_attribute: 1}],
                num_epochs=num_epochs,
                classifier_num_hidden_units=units,
                adversary_loss_weight=weight,
                seed=SEED 
            )
            
            dataset_train_fold = prepare_aif360_data(X_fold_train, y_fold_train_named, protected_attribute)
            
            with SuppressPrints(): # Suprimir prints do AIF360/TensorFlow
                model.fit(dataset_train_fold)
            
            dataset_val_fold = prepare_aif360_data(X_fold_val, y_fold_val_named, protected_attribute)
            y_pred_val = model.predict(dataset_val_fold)
            model.close_session() # Fechar sessão do modelo do fold

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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)

    # --- SEÇÃO DE VISUALIZAÇÃO DO OPTUNA ---
    print("\n--- Gerando Visualizações do Estudo Optuna (InAdversarialDebiasing) ---")
    if len(study.trials) > 0:
        try:
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio CV", "Fairness Score Médio CV"])
            fig_pareto.show()
            # fig_pareto.write_html("InAD_pareto_front.html")

            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # fig_history.write_html("InAD_optimization_history.html")

            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # fig_slice.write_html("InAD_slice_plot.html")
            
            # Para plot_param_importances, precisamos garantir que t.values exista
            valid_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
            if valid_trials_for_importance: # Só plota se houver trials válidos
                fig_importance_f1 = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[0] if t.values is not None and len(t.values) > 0 else float('nan'), 
                    target_name="Importância para F1-score CV"
                )
                fig_importance_f1.show()

                fig_importance_fs = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[1] if t.values is not None and len(t.values) > 1 else float('nan'), 
                    target_name="Importância para Fairness Score CV"
                )
                fig_importance_fs.show()
            else:
                print("Nenhum trial completo com valores válidos para gerar gráficos de importância.")

        except Exception as e:
            print(f"Erro ao gerar visualizações do Optuna: {e}")
            print("Verifique se Plotly está instalado e se há trials suficientes e válidos no estudo.")
    else:
        print("Nenhum trial no estudo para gerar visualizações.")
    # --- FIM DA SEÇÃO DE VISUALIZAÇÃO ---
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")    
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        # Copia os parâmetros originais para retorno ANTES de qualquer modificação (como .pop())
        final_best_params_to_return = best_trial_found.params.copy()
        
        # Trabalha com o dicionário original (ou uma cópia se preferir não modificar best_trial_found.params)
        # para extrair parâmetros específicos do modelo.
        # Para AdversarialDebiasing, best_params SÃO os parâmetros do modelo.
        params_for_model = best_trial_found.params 

        best_score_f1_cv = best_trial_found.user_attrs.get("avg_f1")
        final_di_cv = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal (Baseada na CV) ---")
        if best_score_f1_cv is not None: print(f"F1-score (CV do trial selecionado): {best_score_f1_cv:.4f}")
        if final_di_cv is not None: print(f"Disparate Impact (DI) (CV do trial selecionado): {final_di_cv:.4f}")
        print(f"Melhores Hiperparâmetros (AdversarialDebiasing): {final_best_params_to_return}")
        
        best_model = AdversarialDebiasingWrapper(
            protected_attribute_name=protected_attribute,
            privileged_groups=[{protected_attribute: 0}],
            unprivileged_groups=[{protected_attribute: 1}],
            **params_for_model, # Desempacota os parâmetros otimizados para o wrapper
            seed=SEED
        )
        
        # Garantir nomes e índices para y_train e y_test
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        
        y_test_named = y_test.copy() # Usado para dataset_test, embora y não seja usado no predict
        if y_test_named.name is None: y_test_named.name = label_col_name


        dataset_train_full = prepare_aif360_data(X_train, y_train_named, protected_attribute)
        # Para predição, y não é usado, então podemos passar um dummy ou y_test_named
        dataset_test_for_pred = prepare_aif360_data(X_test, y_test_named, protected_attribute) 
        
        with SuppressPrints():
            best_model.fit(dataset_train_full)
            
        y_pred_train = best_model.predict(dataset_train_full)
        y_pred_test = best_model.predict(dataset_test_for_pred)
        # best_model.close_session() # Movido para depois do LIME

        df_train['source'] = 'train'
        df_train['prediction'] = y_pred_train
        df_test['source'] = 'test'
        df_test['prediction'] = y_pred_test
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        explanations = []
        for i in range(min(10, len(X_test))):
            # LIME espera um array NumPy para data_row
            instance_to_explain = X_test.iloc[i].values
            
            # A função predict_fn do LIME deve aceitar um array NumPy e retornar probabilidades
            exp = explainer.explain_instance(
                data_row=instance_to_explain, 
                # A predict_proba do wrapper já lida com a conversão de ndarray para DataFrame e AIF360 dataset
                predict_fn=best_model.predict_proba, 
                num_features=len(attributes) 
            )
            explanations.append(exp)
        
        best_model.close_session() # Fechar sessão após LIME
        
        return df_final, best_model, final_best_params_to_return, best_score_f1_cv, explanations
    else:
        print(
            "\nERRO no script principal: Não foi possível selecionar um trial com os critérios definidos. "
            "O treinamento otimizado não pode prosseguir."
        )
        return None, None, None, None, None



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Pos-processing Models                        """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def PostEOddsPostprocessingRandomFlorestOptuna(X_train, y_train, X_test, y_test, 
                                               df_train, df_test, 
                                               protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    Equalized Odds Postprocessing (EqOddsPostprocessing) to adjust predictions.
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'
    
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['0', '1'], # Ajuste se suas classes tiverem outros nomes
        discretize_continuous=True,
        random_state=SEED
    )
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
            
        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            y_val_pred = clf_rf.predict(X_fold_val)
            
            # Garantir nomes e índices consistentes para Series de labels
            y_fold_val_named = y_fold_val.copy()
            if not isinstance(y_fold_val_named, pd.Series) or y_fold_val_named.name is None:
                 y_fold_val_named = pd.Series(y_fold_val, name=label_col_name, index=X_fold_val.index)
            elif not y_fold_val_named.index.equals(X_fold_val.index): # Garante alinhamento se já for Series
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
                # print(f"AVISO - Trial {trial.number}, Fold: Instâncias negativas insuficientes...") # Comentado
                pass 
            else:
                try:
                    eq = EqOddsPostprocessing(
                        unprivileged_groups=[{protected_attribute: 1}],
                        privileged_groups=[{protected_attribute: 0}],
                        seed=SEED
                    )
                    eq.fit(bd_true, bd_pred)
                    bd_pred_post = eq.predict(bd_pred)
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
                       "Unable to solve optimization problem" in str(e):
                        # print(f"AVISO - Trial {trial.number}, Fold: Erro no linprog ({e})...") # Comentado
                        pass 
                    else:
                        # print(f"ERRO - Trial {trial.number}, Fold: ValueError inesperado ({e})...") # Comentado
                        pass 
                except ZeroDivisionError as e:
                    # print(f"AVISO - Trial {trial.number}, Fold: ZeroDivisionError ({e})...") # Comentado
                    pass 
            
            f1_scores.append(f1_fold)
            di_scores.append(di_fold)
            
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)
    
    # --- SEÇÃO DE VISUALIZAÇÃO DO OPTUNA ---
    print("\n--- Gerando Visualizações do Estudo Optuna (PostEOddsPostprocessing) ---")
    if len(study.trials) > 0:
        try:
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio CV", "Fairness Score Médio CV"])
            fig_pareto.show()
            # fig_pareto.write_html("PostEOdds_pareto_front.html")

            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # fig_history.write_html("PostEOdds_optimization_history.html")

            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # fig_slice.write_html("PostEOdds_slice_plot.html")

            valid_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
            if valid_trials_for_importance:
                fig_importance_f1 = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[0] if t.values is not None and len(t.values) > 0 else float('nan'), 
                    target_name="Importância para F1-score CV"
                )
                fig_importance_f1.show()

                fig_importance_fs = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[1] if t.values is not None and len(t.values) > 1 else float('nan'), 
                    target_name="Importância para Fairness Score CV"
                )
                fig_importance_fs.show()
            else:
                print("Nenhum trial completo com valores válidos para gerar gráficos de importância.")

        except Exception as e:
            print(f"Erro ao gerar visualizações do Optuna: {e}")
    else:
        print("Nenhum trial no estudo para gerar visualizações.")
    # --- FIM DA SEÇÃO DE VISUALIZAÇÃO ---
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")    
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        best_params = best_trial_found.params # Contém apenas params do RF para este modelo
        best_score = best_trial_found.user_attrs.get("avg_f1")
        final_di = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal ---")
        if best_score is not None: print(f"F1-score (CV do trial selecionado): {best_score:.4f}")
        if final_di is not None: print(f"Disparate Impact (DI) (CV do trial selecionado): {final_di:.4f}")
        print(f"Melhores Hiperparâmetros (RandomForest): {best_params}")
        
        best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        y_train_raw = best_model.predict(X_train)
        y_test_raw  = best_model.predict(X_test)
        
        # Garantir nomes e índices para Series de labels e predições brutas
        y_train_named = y_train.copy()
        if y_train_named.name is None: y_train_named.name = label_col_name
        
        y_test_named = y_test.copy() # Usado para bd_test_pred
        if y_test_named.name is None: y_test_named.name = label_col_name

        y_train_raw_series = pd.Series(y_train_raw, index=y_train_named.index, name=label_col_name)
        y_test_raw_series = pd.Series(y_test_raw, index=y_test_named.index, name=label_col_name)

        bd_train_true = prepare_aif360_data(X_train, y_train_named, protected_attribute)
        bd_train_pred = prepare_aif360_data(X_train, y_train_raw_series, protected_attribute)
        bd_test_pred  = prepare_aif360_data(X_test,  y_test_raw_series, protected_attribute)
        
        eq_odds = EqOddsPostprocessing(
            unprivileged_groups=[{protected_attribute: 1}],
            privileged_groups=[{protected_attribute: 0}],
            seed=SEED
        )
        
        y_train_pred_final = y_train_raw 
        y_test_pred_final = y_test_raw   
        lime_predict_fn = lambda X_lime_input_df: best_model.predict_proba(X_lime_input_df)

        try:
            # print("INFO: Ajustando o EqOddsPostprocessing final no conjunto de treino completo...")
            eq_odds.fit(bd_train_true, bd_train_pred)
            # print("INFO: Aplicando EqOddsPostprocessing final aos dados de treino e teste...")
            y_train_pred_post = eq_odds.predict(bd_train_pred).labels.ravel()
            y_test_pred_post  = eq_odds.predict(bd_test_pred).labels.ravel()
            
            y_train_pred_final = y_train_pred_post 
            y_test_pred_final = y_test_pred_post   

            def post_processed_predict_proba(X_lime_input_np):
                X_lime_df = pd.DataFrame(X_lime_input_np, columns=attributes)
                probs_raw = best_model.predict_proba(X_lime_df)
                labels_raw = np.argmax(probs_raw, axis=1)
                labels_raw_series = pd.Series(labels_raw, index=X_lime_df.index, name=label_col_name)
                
                bd_lime = prepare_aif360_data(X_lime_df, labels_raw_series, protected_attribute)
                bd_lime_post = eq_odds.predict(bd_lime) 
                return np.vstack([1 - bd_lime_post.scores.ravel(), bd_lime_post.scores.ravel()]).T
            
            lime_predict_fn = post_processed_predict_proba
            # print("INFO: Pós-processamento final aplicado com sucesso.")

        except ValueError as e:
            if "c must not contain values inf, nan, or None" in str(e) or \
               "Unable to solve optimization problem" in str(e):
                print(f"AVISO FINAL: Erro no solver ao ajustar EqOdds no dataset completo ({e}). Usando predições brutas.")
            else:
                print(f"ERRO FINAL: ValueError inesperado durante EqOdds final ({e}). Usando predições brutas.")
        except ZeroDivisionError as e:
            print(f"AVISO FINAL: ZeroDivisionError durante EqOdds final ({e}). Usando predições brutas.")
        
        df_train['source'] = 'train'
        df_train['prediction'] = y_train_pred_final
        df_test['source'] = 'test'
        df_test['prediction'] = y_test_pred_final
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        explanations = [
            explainer.explain_instance(
                X_test.iloc[i].values,
                lime_predict_fn,
                num_features=len(attributes) # Usar len(attributes) para consistência
            ) for i in range(min(10, len(X_test)))
        ]
        
        return df_final, best_model, best_params, best_score, explanations
    else:
        print("\nERRO: Não foi possível selecionar um trial com os critérios definidos...")
        return None, None, None, None, None
    


def PostCalibratedEOddsRandomFlorestOptuna(X_train, y_train, X_test, y_test, 
                                           df_train, df_test, 
                                           protected_attribute, num_trials=50):
    """
    This function trains and optimizes a RandomForest model, and then applies 
    Calibrated Equalized Odds Postprocessing (CalibratedEqOddsPostprocessing) 
    to adjust predictions on both the training and test sets,
    in order to reduce bias between privileged and underprivileged groups.
    """
    
    attributes = X_train.columns.tolist()
    label_col_name = y_train.name if y_train.name is not None else 'Target'
    
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(), # Correto, usa as features originais de X_train
        class_names=['0', '1'], 
        discretize_continuous=True,
        random_state=SEED
    )
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap_enabled = trial.suggest_categorical('bootstrap', [True, False])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])
        
        if bootstrap_enabled:
            max_samples_val = trial.suggest_float('max_samples', 0.6, 1.0, step=0.05)
        else:
            max_samples_val = None
            
        cost_constraint = trial.suggest_categorical("cost_constraint", ["fnr", "fpr", "weighted"])

        clf_rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, bootstrap=bootstrap_enabled,
            class_weight=class_weight, max_samples=max_samples_val,
            n_jobs=1, random_state=SEED
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        f1_scores = []
        di_scores = []
    
        for train_index, val_index in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
            
            clf_rf.fit(X_fold_train, y_fold_train)
            y_val_pred = clf_rf.predict(X_fold_val)
            
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
                # print(f"AVISO - Trial {trial.number}, Fold (Cost: {cost_constraint}): Instâncias negativas insuficientes...") # Comentado
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
                        # print(f"AVISO - Trial {trial.number}, Fold (Cost: {cost_constraint}): Erro no solver ({e}).") # Comentado
                        pass 
                    else:
                        # print(f"ERRO - Trial {trial.number}, Fold (Cost: {cost_constraint}): ValueError inesperado ({e}).") # Comentado
                        pass 
                except ZeroDivisionError as e:
                    # print(f"AVISO - Trial {trial.number}, Fold (Cost: {cost_constraint}): ZeroDivisionError ({e}).") # Comentado
                    pass 
            
            f1_scores.append(f1_fold)
            di_scores.append(di_fold)
            
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        avg_di = np.mean(di_scores) if di_scores else 0.0
        fairness_score = 1.0 - abs(avg_di - 1.0)
        
        trial.set_user_attr("avg_f1", avg_f1)
        trial.set_user_attr("avg_di", avg_di)
        
        return avg_f1, fairness_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study  = optuna.create_study(
        directions=['maximize', 'maximize'],
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=num_trials)
    
    # --- SEÇÃO DE VISUALIZAÇÃO DO OPTUNA ---
    print("\n--- Gerando Visualizações do Estudo Optuna (PostCalibratedEOdds) ---")
    if len(study.trials) > 0:
        try:
            fig_pareto = optuna.visualization.plot_pareto_front(study, target_names=["F1-score Médio CV", "Fairness Score Médio CV"])
            fig_pareto.show()
            # fig_pareto.write_html("PostCalibEOdds_pareto_front.html") # Nome de arquivo específico

            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.show()
            # fig_history.write_html("PostCalibEOdds_optimization_history.html")

            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.show()
            # fig_slice.write_html("PostCalibEOdds_slice_plot.html")

            valid_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
            if valid_trials_for_importance:
                fig_importance_f1 = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[0] if t.values is not None and len(t.values) > 0 else float('nan'), 
                    target_name="Importância para F1-score CV"
                )
                fig_importance_f1.show()

                fig_importance_fs = optuna.visualization.plot_param_importances(
                    study, 
                    target=lambda t: t.values[1] if t.values is not None and len(t.values) > 1 else float('nan'), 
                    target_name="Importância para Fairness Score CV"
                )
                fig_importance_fs.show()
            else:
                print("Nenhum trial completo com valores válidos para gerar gráficos de importância.")

        except Exception as e:
            print(f"Erro ao gerar visualizações do Optuna: {e}")
    else:
        print("Nenhum trial no estudo para gerar visualizações.")
    # --- FIM DA SEÇÃO DE VISUALIZAÇÃO ---
    
    print("\n--- Iniciando Seleção Customizada do Melhor Trial ---")    
    best_trial_found = select_best_hyperparameters_trial(study)
    
    if best_trial_found:
        final_best_params_to_return = best_trial_found.params.copy()
        
        params_for_model_instantiation = best_trial_found.params # Isso será modificado pelo .pop()
        cost_constraint_best = params_for_model_instantiation.pop('cost_constraint')
        rf_params_best = params_for_model_instantiation # Contém apenas params do RF
        
        best_score_cv = best_trial_found.user_attrs.get("avg_f1")
        final_di_cv = best_trial_found.user_attrs.get("avg_di")
    
        print("\n--- Configuração Final Selecionada Pelo Script Principal (Baseada na CV) ---")
        if best_score_cv is not None: print(f"F1-score (CV do trial selecionado): {best_score_cv:.4f}")
        if final_di_cv is not None: print(f"Disparate Impact (DI) (CV do trial selecionado): {final_di_cv:.4f}")
        print(f"Melhor cost_constraint (CV): {cost_constraint_best}") # Log do cost_constraint
        print(f"Melhores Hiperparâmetros RF (CV): {rf_params_best}")
        # print(f"Todos Hiperparâmetros Otimizados: {final_best_params_to_return}") # Opcional

        best_model = RandomForestClassifier(**rf_params_best, random_state=SEED, n_jobs=1)
        best_model.fit(X_train, y_train)
        
        y_train_raw = best_model.predict(X_train)
        y_test_raw  = best_model.predict(X_test)
        
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
            cost_constraint=cost_constraint_best, # Usar o otimizado
            seed=SEED,
        )
        
        y_train_pred_final = y_train_raw 
        y_test_pred_final = y_test_raw   
        lime_predict_fn = lambda X_lime_input_df: best_model.predict_proba(X_lime_input_df) 

        try:
            # print("INFO: Ajustando o CalibratedEqOddsPostprocessing final...") # Comentado
            final_post.fit(bd_train_true, bd_train_pred)
            # print("INFO: Aplicando CalibratedEqOddsPostprocessing final...") # Comentado
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
            # print("INFO: Pós-processamento final aplicado com sucesso.") # Comentado

        except ValueError as e:
            if "c must not contain values inf, nan, or None" in str(e) or \
               "Unable to solve optimization problem" in str(e) or \
               "constraint violation" in str(e).lower():
                # print(f"AVISO FINAL: Erro no solver ao ajustar CalibratedEqOdds ({e}). Usando predições brutas.") # Comentado
                pass 
            else:
                # print(f"ERRO FINAL: ValueError inesperado durante CalibratedEqOdds ({e}). Usando predições brutas.") # Comentado
                pass 
        except ZeroDivisionError as e:
            # print(f"AVISO FINAL: ZeroDivisionError durante CalibratedEqOdds ({e}). Usando predições brutas.") # Comentado
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
                num_features=len(attributes) 
            ) for i in range(min(10, len(X_test)))
        ]
        
        # Retorna o dicionário completo de parâmetros que foi otimizado
        return df_final, best_model, final_best_params_to_return, best_score_cv, explanations 
    else:
        print("\nERRO: Não foi possível selecionar um trial com os critérios definidos...")
        return None, None, None, None, None




