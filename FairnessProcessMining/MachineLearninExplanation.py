import numpy as np
import pandas as pd

def FeatureImportance(model, columns, label):
    """
    This function calculates the importances of the variables (features) of a trained model and organizes them in a DataFrame
    for easy visualization. The importances are obtained from the model's `feature_importances_` attribute, which
    is common in tree-based models such as RandomForest and XGBoost. The resulting DataFrame contains the variables
    and their respective importances, and can be labeled to identify the origin or type of model.
    
    Parameters:
    - model: The trained model that has the `feature_importances_` attribute.
    - columns (list): List with the names of the variables used in the model training.
    - label (str): Label to identify the set of importances, usually the model name or a description.
    
    Returns:
    - df_importances (pd.DataFrame): DataFrame containing the variables and their respective importances.
    """
    # Get the importances of the model features
    importances = model.feature_importances_
    
    # Create a DataFrame to visualize the importances
    df_importances = pd.DataFrame({'Feature': columns, f'Importance ({label})': importances})
    
    return df_importances


def FeatureImportanceLime(explanations, feature_names):
    """
    This function calculates the average importance of variables (features) from the LIME explanations generated
    for a set of instances. The importance of variables is obtained by summing the local importances
    provided by LIME for each explanation and then dividing by the total number of explanations, resulting in an average importance for each variable.
    
    Parameters:
    - explanations (list): List of LIME explanations for different instances. Each explanation contains the local importances of variables for the target class.
    - feature_names (list): List containing the names of the variables.
    
    Returns:
    - feature_importances (numpy.ndarray): Array containing the average importance of each variable.
    """
    feature_importances = np.zeros(len(feature_names))
    for explanation in explanations:
        for feature, importance in explanation.local_exp[1]:  # 1 é a classe alvo
            feature_importances[feature] += importance
    feature_importances /= len(explanations)
    return feature_importances