import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""    Identify and eliminate duplicate records in the database              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def KeepFirstDuplicateCases(df, variaveis, var_timestamp, var_target, var_id, var_date):
    """
    This function identifies and removes duplicate cases in a DataFrame based on a list of specified variables.
    Duplicate cases are identified by considering all variables of interest, including the Target, and only
    the first record for each set of duplicates is kept. The function returns three DataFrames:
    1. Duplicate records that were removed.
    2. The original DataFrame.
    3. The final DataFrame with the duplicates removed.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be processed.
    - variables (list): List of variables of interest, including the protected attribute and other features.
    - var_timestamp (str): Name of the timestamp column (default: 'time:timestamp').
    - var_target (str): Name of the target column (default: 'Target').
    - var_id (str): Name of the case identifier column (default: 'case:case').
    - var_date (str): Name of the case end date column (default: 'case_end_date').

    Returns:
    - pd.DataFrame: Duplicate records that were removed.
    - pd.DataFrame: The original DataFrame.
    - pd.DataFrame: The final DataFrame with duplicates removed.
    """
    # Convert 'time:timestamp' to datetime if it is in the variable list
    if var_timestamp in df.columns and var_timestamp in variaveis:
        df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Attributes present in the DataFrame
    variaveis_df = [var for var in variaveis if var in df.columns]

    # Remove 'Target' from the list of variables to group by
    variaveis_sem_target = [var for var in variaveis_df if var != var_target]

    # Include 'Target' in the list of variables to consider in duplicates
    variaveis_com_target = variaveis_sem_target + [var_target]

    # Group by all variables of interest, including 'Target'
    df_selected = df[variaveis_com_target + [var_date]]

    # Sort by 'case_end_date' to ensure the first record is at the end
    df_selected = df_selected.sort_values(var_date)

    # Identify duplicates based on attributes of interest and 'Target', keeping the first record
    df_last_records = df_selected.drop_duplicates(subset=variaveis_com_target, keep='first')

    # Find the duplicate records (those that were removed by drop_duplicates)
    duplicates = df_selected[~df_selected.index.isin(df_last_records.index)]

    # Delete all duplicate records from the original database
    df_cleaned = df.loc[df_last_records.index]

    return duplicates, df, df_cleaned


def KeepLastDuplicateCases(df, variaveis, var_timestamp, var_target, var_id, var_date):
    """
    This function identifies and removes duplicate cases in a DataFrame based on a list of specified variables.
    Duplicate cases are identified by considering all variables of interest, including the Target, and only
    the last record for each set of duplicates is kept. The function returns three DataFrames:
    1. Duplicate records that were removed.
    2. The original DataFrame.
    3. The final DataFrame with the duplicates removed.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be processed.
    - variables (list): List of variables of interest, including the protected attribute and other features.
    - var_timestamp (str): Name of the timestamp column (default: 'time:timestamp').
    - var_target (str): Name of the target column (default: 'Target').
    - var_id (str): Name of the case identifier column (default: 'case:case').
    - var_date (str): Name of the case end date column (default: 'case_end_date').

    Returns:
    - pd.DataFrame: Duplicate records that were removed.
    - pd.DataFrame: The original DataFrame.
    - pd.DataFrame: The final DataFrame with duplicates removed.
    """
    
    # Convert 'time:timestamp' to datetime if it is in the variable list
    if var_timestamp in df.columns and var_timestamp in variaveis:
        df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Attributes present in the DataFrame
    variaveis_df = [var for var in variaveis if var in df.columns]

    # Remove 'Target' from the list of attributes to group by
    variaveis_sem_target = [var for var in variaveis_df if var != var_target]

    # Include 'Target' in the list of attributes to consider in duplicates
    variaveis_com_target = variaveis_sem_target + [var_target]

    # Select the columns of interest and 'case_end_date'
    df_selected = df[variaveis_com_target + [var_date]]

    # Sort by 'case_end_date' to ensure the last record is at the end
    df_selected = df_selected.sort_values(var_date)

    # Identify duplicates based on attributes of interest and 'Target', keeping the last record
    df_last_records = df_selected.drop_duplicates(subset=variaveis_com_target, keep='last')

    # Find the duplicate records (those that were removed by drop_duplicates)
    duplicates = df_selected[~df_selected.index.isin(df_last_records.index)]

    # Delete all duplicate records from the original database
    df_cleaned = df.loc[df_last_records.index]

    return duplicates, df, df_cleaned


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""     Identify and eliminate conflicting records in the database           """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def KeepFirstConflictingCases(df, variaveis, var_timestamp='time:timestamp', var_target='Target', var_id='case:case', var_date='case_end_date'):
    """
    This function handles conflicting cases in a DataFrame. It identifies records that have 
    conflicting values ​​for a specified list of variables and keeps only the first record 
    of each set of conflicts, based on the smallest case end date (case_end_date). 
    The function returns three DataFrames: 
    1. Conflicting groups with more than one Target value. 
    2. Identified conflicting cases. 
    3. Final DataFrame with resolved conflicts, keeping only the first record of each conflicting group.
  
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be processed.
    - variables (list): List of variables of interest, including the protected attribute and other features.
    - var_timestamp (str): Name of the timestamp column (default: 'time:timestamp').
    - var_target (str): Name of the target column (default: 'Target').
    - var_id (str): Name of the case identifier column (default: 'case:case').
    - var_date (str): Name of the case end date column (default: 'case_end_date').

    Returns:
    - pd.DataFrame: Conflicting groups with more than one Target value.
    - pd.DataFrame: Conflicting cases identified.
    - pd.DataFrame: Final DataFrame with resolved conflicts.
    """
      
    # Convert 'time:timestamp' to datetime if it is in the variable list
    if var_timestamp in df.columns and var_timestamp in variaveis:
        df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Attributes present in the DataFrame
    variaveis_df = [var for var in variaveis if var in df.columns]

    # Remove 'Target' from the list of attributes to group by
    variaveis_sem_target = [var for var in variaveis_df if var != var_target]

    # Group by all variables except 'Target' and count the number of unique Targets
    grouped = df.groupby(variaveis_sem_target)[var_target].nunique().reset_index().copy()

    # Rename the count column to 'unique_targets'
    grouped = grouped.rename(columns={var_target: 'unique_targets'})

    # Filter groups that have more than one different 'Target'
    conflicting_groups = grouped[grouped['unique_targets'] > 1]

    if conflicting_groups.empty:
        return pd.DataFrame(), pd.DataFrame(), df  # If there are no conflicts, return the original df

    # Identify conflicting case records
    conflicting_cases = df.merge(conflicting_groups[variaveis_sem_target], on=variaveis_sem_target)

    # Keep only the first record of each set of conflicting features, based on 'case_end_date'
    df_first_records = conflicting_cases.sort_values(var_date).groupby(variaveis_sem_target, as_index=False).first()

    # Remove conflicting records from the original DataFrame
    df_cleaned = df[~df[var_id].isin(conflicting_cases[var_id])].copy()  # Using .copy() to avoid fragmentation

    # Concatenate the clean DataFrame with the first records of the conflicting groups
    df_final = pd.concat([df_cleaned, df_first_records], ignore_index=True)

    return conflicting_groups, conflicting_cases, df_final



def KeepLastConflictingCases(df, variaveis, var_timestamp, var_target, var_id, var_date):
    """
    This function handles conflicting cases in a DataFrame. It identifies records that have 
    conflicting values ​​for a specified list of variables and keeps only the last record of 
    each set of conflicts, based on the largest case end date (case_end_date). 
    The function returns three DataFrames: 
    1. Conflicting groups with more than one Target value. 
    2. Identified conflicting cases. 
    3. Final DataFrame with resolved conflicts, keeping only the last record of each conflicting group.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be processed.
    - variables (list): List of variables of interest, including the protected attribute and other features.
    - var_timestamp (str): Name of the timestamp column (default: 'time:timestamp').
    - var_target (str): Name of the target column (default: 'Target').
    - var_id (str): Name of the case identifier column (default: 'case:case').
    - var_date (str): Name of the case end date column (default: 'case_end_date').

    Returns:
    - pd.DataFrame: Conflicting groups with more than one Target value.
    - pd.DataFrame: Conflicting cases identified.
    - pd.DataFrame: Final DataFrame with resolved conflicts.
    """
    
    # Convert 'time:timestamp' to datetime if it is in the variable list
    if var_timestamp in df.columns and var_timestamp in variaveis:
        df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Attributes present in the DataFrame
    variaveis_df = [var for var in variaveis if var in df.columns]

    # Remove 'Target' from the list of attributes to group by
    variaveis_sem_target = [var for var in variaveis_df if var != var_target]

    # Group by all variables except 'Target' and count the number of unique Targets
    grouped = df.groupby(variaveis_sem_target)[var_target].nunique().reset_index().copy()

    # Rename the count column to 'unique_targets'
    grouped = grouped.rename(columns={var_target: 'unique_targets'})

    # Filter groups that have more than one different 'Target'
    conflicting_groups = grouped[grouped['unique_targets'] > 1]

    if conflicting_groups.empty:
        return pd.DataFrame(), pd.DataFrame(), df  # If there are no conflicts, return the original df

    # Identify conflicting case records
    conflicting_cases = df.merge(conflicting_groups[variaveis_sem_target], on=variaveis_sem_target)

    # Keep only the last record of each set of conflicting features, based on 'case_end_date'
    df_last_records = conflicting_cases.sort_values(var_date).groupby(variaveis_sem_target, as_index=False).last()

    # Remove conflicting records from the original DataFrame
    df_cleaned = df[~df[var_id].isin(conflicting_cases[var_id])].copy()  # Using .copy() to avoid fragmentation

    # Concatenate the clean DataFrame with the first records of the conflicting groups
    df_final = pd.concat([df_cleaned, df_last_records], ignore_index=True)

    return conflicting_groups, conflicting_cases, df_final



def RemoveAllConflictingCases(df, variaveis, var_timestamp, var_target, var_id):
    """   
    This function handles conflicting cases in a DataFrame. It identifies records that have 
    conflicting values ​​for a specified list of variables and completely removes all conflicting records.
    The function returns three DataFrames: 
    1. Conflicting groups with more than one Target value. 
    2. Identified conflicting cases. 
    3. Final DataFrame with resolved conflicts, keeping only the last record of each conflicting group.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados a serem processados.
    - variaveis (list): Lista de variáveis de interesse, incluindo atributos e features.
    - var_timestamp (str): Nome da coluna de timestamp (padrão: 'time:timestamp').
    - var_target (str): Nome da coluna de target (padrão: 'Target').
    - var_id (str): Nome da coluna de identificador de caso (padrão: 'case:case').

    Retorna:
    - pd.DataFrame: Grupos conflitantes com mais de um valor de Target.
    - pd.DataFrame: Casos conflitantes identificados.
    - pd.DataFrame: DataFrame final sem nenhuma instância conflitante.
    """

    # Convert 'time:timestamp' to datetime if it is in the variable list
    if var_timestamp in df.columns and var_timestamp in variaveis:
        df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Attributes present in the DataFrame
    variaveis_df = [v for v in variaveis if v in df.columns]

    # Remove 'Target' from the list of attributes to group by
    vars_sem_target = [v for v in variaveis_df if v != var_target]

    # Count how many distinct targets each combination of vars_sem_target has
    counts = (
        df
        .groupby(vars_sem_target)[var_target]
        .nunique()
        .reset_index()
        .rename(columns={var_target: 'unique_targets'})
    )

    # Identify groups with more than one distinct target
    conflicting_groups = counts[counts['unique_targets'] > 1]

    if conflicting_groups.empty:
        # No conflicts: returns empty for groups and cases, in addition to the original df
        return pd.DataFrame(), pd.DataFrame(), df.copy()

    # Find all records that belong to these conflicting groups
    conflicting_cases = df.merge(conflicting_groups[vars_sem_target], on=vars_sem_target)

    # Remove **all** conflicting instances from the original DataFrame
    df_clean = df[~df[var_id].isin(conflicting_cases[var_id])].copy()

    return conflicting_groups, conflicting_cases, df_clean


