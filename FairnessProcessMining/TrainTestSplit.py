import pandas as pd
from sklearn.model_selection import train_test_split
from FairnessProcessMining import Metrics

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                        Auxiliary functions                                """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def PrepareData(df, var_date):
    """
    Converts a date column to datetime format and sorts the DataFrame by that column.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the data.
    - var_date(str): Name of the column containing the case end dates (default: 'case_end_date').
    
    Returns:
    - pd.DataFrame: DataFrame sorted by the specified date column.
    """
    
    df[var_date] = pd.to_datetime(df[var_date])
    df.sort_values(var_date, inplace=True)
    return df


def PrepareStratificationColumn(df, var_protect, var_target):
    """
    Creates a stratification column based on the interaction between a protected attribute and the target.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - var_protect (str): Name of the column containing the protected attribute (default: 'case:protected').
    - var_target (str): Name of the column containing the target (default: 'Target').
    
    Returns:
    - pd.DataFrame: DataFrame with the new stratification column.
    """
    
    df['stratify_col'] = df[var_protect].astype(str) + "_" + df[var_target].astype(str)
    return df


def TrainTestSplitTemporal(df, test_size):
    """
    Splits the data keeping the temporal order, without stratification.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - test_size (float): Proportion of the data to be used as the test set.
    
    Returns:
    - pd.DataFrame: Training set.
    - pd.DataFrame: Test set.
    """
    
    cutoff_index = int((1 - test_size) * len(df))
    df_train = df.iloc[:cutoff_index]
    df_test = df.iloc[cutoff_index:]
    return df_train, df_test


def TrainTestSplitRandom(df, test_size, stratify_col=None, random_state=42):
    """
    Randomly splits the data, with the option to stratify based on a column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - test_size (float): Proportion of the data to be used as the test set (default: 0.3).
    - stratify_col (str): Name of the column used for stratification (optional).
    - random_state (int): Randomization seed for reproducibility (default: 42).
    
    Returns:
    - pd.DataFrame: Training set.
    - pd.DataFrame: Test set.
    """
    
    if stratify_col and stratify_col in df.columns:
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=random_state)
    else:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test


def TrainTestSplitTemporalStratified(df, test_size, stratify_col=None):
    """
    Splits the data keeping the temporal order, with the option to stratify based on a column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - test_size (float): Proportion of the data to be used as the test set (default: 0.3).
    - stratify_col (str): Name of the column used for stratification (optional).
    
    Returns:
    - pd.DataFrame: Training set.
    - pd.DataFrame: Test set.
    """
    
    cutoff_index = int((1 - test_size) * len(df))
    if stratify_col and stratify_col in df.columns:
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[stratify_col])
    else:
        df_train = df.iloc[:cutoff_index]
        df_test = df.iloc[cutoff_index:]
    return df_train, df_test


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                        Main functions                                    """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def SplitDataTemporal(df, test_size, var_date):
    """
    Splits the data into training and testing data considering the case closing date.
    
    The division is temporal, maintaining the order of the cases according to time.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - test_size (float): Proportion of the data to be used as the test set.
    - var_date (str): Name of the column containing the case closing dates.
    
    Returns:
    - pd.DataFrame: Training set.
    - pd.DataFrame: Test set.
    """
    
    df = PrepareData(df, var_date)
    df_train, df_test = TrainTestSplitTemporal(df, test_size)
    return df_train, df_test


def SplitDataRandom(df, test_size, var_protect, var_target, random_state=42):
    """
    Randomly divides the data into training and testing, with subsequent rebalancing, considering
    the target and the protected attribute.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - test_size (float): Proportion of the data to be used as the test set (default: 0.3).
    - random_state (int): Randomization seed for reproducibility (default: 42).
    - var_protect (str): Name of the column containing the protected attribute (default: 'case:protected').
    - var_target (str): Name of the column containing the target (default: 'Target').
    
    Returns:
    - pd.DataFrame: Training set.
    - pd.DataFrame: Test set.
    """
    
    df = PrepareStratificationColumn(df, var_protect, var_target)
    df_train, df_test = TrainTestSplitRandom(df, test_size, stratify_col='stratify_col', random_state=random_state)
    return df_train, df_test


def SplitDataTemporalStratified(df, test_size, var_protect, var_target, var_date, random_state=42):
    """
    Splits the data into training and testing based on the case end date.
    
    The split can be stratified based on a stratification column created from the target and the protected attribute.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - test_size (float): Proportion of the data to be used as the test set (default: 0.3).
    - random_state (int): Randomization seed for reproducibility (default: 42).
    - var_protect (str): Name of the column containing the protected attribute (default: 'case:protected').
    - var_target (str): Name of the column containing the target (default: 'Target').
    - var_date (str): Name of the column containing the case end dates (default: 'case_end_date').
    
    Returns:
    - pd.DataFrame: Training set.
    - pd.DataFrame: Data set test. 
    """
    
    df = PrepareData(df, var_date)
    df = PrepareStratificationColumn(df, var_protect, var_target)
    df_train, df_test = TrainTestSplitTemporalStratified(df, test_size, stratify_col='stratify_col')
    return df_train, df_test


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""             Descriptive Functions of Training/Test Bases                 """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def DescriptiveTrainTest(df_train, df_test, level, outcome_attribute, protected_attribute, privileged_group, unprivileged_group):
    """
    This function calculates the percentages of different attributes in two DataFrames (training and testing) and returns a summary table.
    The percentages include the proportion of target values, the proportion of protected individuals, and the proportion
    of protected individuals who also have the target value. It also calculates the Disparate Impact.
    
    Parameters:
    - df_train(pd.DataFrame): Training DataFrame.
    - df_test(pd.DataFrame): Test DataFrame.
    - outcome_attribute(str): Name of the column containing the target value.
    - protected_attribute(str): Name of the column containing the protected attribute.
    - privileged_group(int/float): Value identifying the privileged group.
    - unprivileged_group(int/float): Value identifying the unprivileged group.
    - level(str): Description of the level (e.g. 'Model' or 'Segment').
    
    Returns:
    - pd.DataFrame: Summary table containing the percentages and Disparate Impact for the training and test sets.
    """
    
    def calculate_percentages(df):
        total_count = len(df)
        target_count_0 = (df[outcome_attribute] == 0).sum()
        target_count_1 = (df[outcome_attribute] == 1).sum()
        protected_count = (df[protected_attribute] == 1).sum()
        protected_target_count = df[(df[outcome_attribute] == 1) & (df[protected_attribute] == 1)].shape[0]
        protected_target0_count = df[(df[outcome_attribute] == 0) & (df[protected_attribute] == 1)].shape[0]
        unprotected_target_count = df[(df[outcome_attribute] == 1) & (df[protected_attribute] == 0)].shape[0]
        unprotected_target0_count = df[(df[outcome_attribute] == 0) & (df[protected_attribute] == 0)].shape[0]
        di = Metrics.DisparateImpact(df, protected_attribute, privileged_group, unprivileged_group, outcome_attribute)
        pos_rate, dp = Metrics.DemographicParity(df, protected_attribute, outcome_attribute)
        
        return {
            'Total': total_count,
            '% Target = 0': "{:.2f}%".format((target_count_0 / total_count) * 100),
            '% Target = 1': "{:.2f}%".format((target_count_1 / total_count) * 100),
            '% Protected = 1': "{:.2f}%".format((protected_count / total_count) * 100),
            '% Protected = 1 & Target = 1': "{:.2f}%".format((protected_target_count / total_count) * 100),
            '% Protected = 1 & Target = 0': "{:.2f}%".format((protected_target0_count / total_count) * 100),
            '% Protected = 0 & Target = 1': "{:.2f}%".format((unprotected_target_count / total_count) * 100),
            '% Protected = 0 & Target = 0': "{:.2f}%".format((unprotected_target0_count / total_count) * 100),
            'Disparate Impact': "{:.2f}".format(di),
            'Demographic Parity': "{:.2f}".format(dp)
        }

    # Calculate statistics for training and testing
    train_stats = calculate_percentages(df_train)
    test_stats = calculate_percentages(df_test)
    
    # Create a DataFrame from the results
    df_summary = pd.DataFrame({
        f'{level} - Train': train_stats,
        f'{level} - Test': test_stats
    })
    
    return df_summary



def Descriptive(df, level, outcome_attribute, protected_attribute, privileged_group, unprivileged_group):
    """
    This function calculates the percentages of different attributes in a single DataFrame,
    without separating it into training and testing. The percentages include the proportion of target values,
    the proportion of protected individuals, the proportion of protected individuals who also have the target value,
    in addition to the Disparate Impact.
    
    Parameters:
    - df (pd.DataFrame): DataFrame of data (all records).
    - outcome_attribute (str): Name of the column containing the target value.
    - protected_attribute (str): Name of the column containing the protected attribute.
    - privileged_group (int/float): Value that identifies the privileged group.
    - unprivileged_group (int/float): Value that identifies the unprivileged group.
    - level (str): Description to identify the set (e.g. 'Complete').
    
    Returns:
    - pd.DataFrame: Summary table containing the percentages and Disparate Impact for the entire set.
    """

    def calculate_percentages(df_local):
        total_count = len(df_local)
        target_count_0 = (df_local[outcome_attribute] == 0).sum()
        target_count_1 = (df_local[outcome_attribute] == 1).sum()
        protected_count = (df_local[protected_attribute] == 1).sum()
        protected_target_count = df_local[(df_local[outcome_attribute] == 1) & (df_local[protected_attribute] == 1)].shape[0]

        # Chamada de alguma função que calcule Disparate Impact
        di = Metrics.DisparateImpact(
            df_local,
            protected_attribute,
            privileged_group,
            unprivileged_group,
            outcome_attribute
        )
        
        pos_rate, dp = Metrics.DemographicParity(df_local, protected_attribute, outcome_attribute)
        
        return {
            'Total': total_count,
            'Total Target = 0': target_count_0,
            'Total Target = 1': target_count_1,
            '% Protected = 1': "{:.2f}%".format((protected_count / total_count) * 100),
            '% Protected = 1 & Target = 1': "{:.2f}%".format((protected_target_count / total_count) * 100),
            'Disparate Impact': "{:.2f}".format(di),
            'Demographic Parity': "{:.2f}".format(dp)
        }

    # Compute statistics for the entire DataFrame
    df = df.drop_duplicates(['case:case'])
    stats = calculate_percentages(df)

    # Construct the output DataFrame (a single column, with the title indicated by 'level')
    df_summary = pd.DataFrame({f'{level}': stats})

    return df_summary

