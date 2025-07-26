import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Activities                                   """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ActivityOneHotEncoding(df, var_id, var_activity):
    """
    This function performs one-hot encoding for activities in a process.
    
    Given a DataFrame containing case and activity identifiers, the function
    creates binary columns indicating the presence or absence of each activity
    within each case. The new columns are concatenated to the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers.
    - var_activity(str): Name of the column containing the activities.
    
    Returns:
    - pd.DataFrame: Original DataFrame concatenated with the one-hot encoding columns.
    """
    
    # Group by case_id and activity, checking for presence
    presence = df.groupby([var_id, var_activity]).size().unstack(fill_value=0).astype(bool).astype(int)
    
    # Concatenate the original data with presence encoding
    final_df = pd.merge(df, presence, on=var_id, how='left')

    return final_df


def ActivityFrequencyEncoding(df, var_id, var_activity):
    """
    This function performs frequency encoding for activities in a process.
    
    Given a DataFrame containing case and activity identifiers, the function
    creates columns indicating the frequency of each activity within each case.
    
    The new frequency columns are then concatenated to the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers (default: 'case:case').
    - var_activity(str): Name of the column containing the activities (default: 'activity').
    
    Returns:
    - pd.DataFrame: Original DataFrame concatenated with the frequency encoding columns.
    """

    # Group by case_id and activity, counting frequency
    freq_counts = df.groupby([var_id, var_activity]).size().unstack(fill_value=0)

    # Concatenate the original data with frequency encoding
    final_df = pd.merge(df, freq_counts, on=var_id, how='left')

    return final_df



def ActivityTfIdfEncoding(df, var_id, var_activity, var_timestamp):
    """
    This function preprocesses the log data, removing the last activity from each trace, and then
    computes the TF-IDF representation for the remaining activities.
    
    TF: counts how many times an activity (term) occurs in a trace (document);
    IDF: checks in how many traces (documents) a given activity (term) occurs in the entire event log.
    TF-IDF: TF x IDF
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the event log.
    - var_id (str): Name of the column containing the case identifiers (default: 'case:concept:name').
    - var_activity (str): Name of the column containing the activities (default: 'concept:name').
    - var_timestamp (str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame containing the TF-IDF matrix of the remaining activities after removing the last activity from each trace.
    - pd.DataFrame: DataFrame with TF (Term Frequency) values ​​for each document and term.
    - pd.DataFrame: DataFrame with IDF (Inverse Document Frequency) values ​​for each term.
    """
        
    # Preprocessing: Replace spaces with underscores in activities
    df_processed = df.copy()
    df_processed[var_activity] = df_processed[var_activity].str.replace(' ', '_')
    df_processed[var_activity] = df_processed[var_activity].str.replace('-', '_')

    # Convert timestamp column to datetime (if applicable)
    df_processed[var_timestamp] = pd.to_datetime(df_processed[var_timestamp])

    # Sort by case ID and timestamp to ensure correct order of activities
    df_sorted = df_processed.sort_values(by=[var_id, var_timestamp])

    # Mark the last activity of each case (trace)
    df_sorted['is_last'] = df_sorted.groupby(var_id).cumcount(ascending=False) == 0

    # Remove the last activity from each trace
    df_without_last = df_sorted[~df_sorted['is_last']].copy()

    # Concatenate all remaining activities from each trace as a single string
    traces = df_without_last.groupby(var_id)[var_activity].apply(lambda x: ' '.join(x)).reset_index()

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()

    # Adjust and transform the remaining activities to generate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(traces[var_activity])

    # Get the names of the activities (features)
    feature_names = vectorizer.get_feature_names_out()

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=traces[var_id])

    # Get IDF values
    idf_values = vectorizer.idf_
    idf_df = pd.DataFrame(idf_values, index=feature_names, columns=['IDF'])

    # Calculate TF manually (term frequency divided by total number of words in each document)
    tf_matrix = vectorizer.transform(traces[var_activity]).toarray()
    tf_df = pd.DataFrame(tf_matrix, columns=feature_names, index=traces[var_id])

    return tfidf_df, tf_df, idf_df



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                    Transition between Activities                         """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def ActivityTransitionsOneHotEncoding(df, var_id, var_activity, var_timestamp):
    """
    This function performs one-hot encoding for activity transitions in a process.
    
    Given a DataFrame containing case identifiers, activities, and timestamps,
    the function identifies the transitions from one activity to another within each case
    and creates a presence matrix of these transitions for each case. In addition,
    the function returns the total of each transition in the original DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - var_id (str): Name of the column containing the case identifiers (default: 'case:case').
    - var_activity (str): Name of the column containing the activities (default: 'activity').
    - timestamp_col (str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame with the presence matrix of transitions per case. - pd.DataFrame: DataFrame with the total of each transition in the original DataFrame.
    """

    # Convert timestamp column to datetime
    df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Sort by case and timestamp
    df = df.sort_values(by=[var_id, var_timestamp])

    # Identify the next activity for each event
    df['next_activity'] = df.groupby(var_id)[var_activity].shift(-1)

    # Create the transitions column
    df['transition'] = df[var_activity] + ' -> ' + df['next_activity']

    # Remove transitions where 'next_activity' is NaN
    df = df.dropna(subset=['next_activity'])
    
    # Delete the last transition of each case
    df['is_last'] = df.groupby(var_id).cumcount(ascending=False) == 0
    df = df[~df['is_last']].copy()

    # Mark the presence of transitions by case
    transition_presence = df.groupby([var_id, 'transition']).size().reset_index(name='count')
    transition_presence['count'] = 1  # Substituir a contagem por 1 para marcar a presença

    # Pivot the data to create the transition presence matrix
    transition_pivot = transition_presence.pivot(index=var_id, columns='transition', values='count').fillna(0)

    # Reset the index so that var_id becomes a column
    transition_pivot = transition_pivot.reset_index()

    # Count the total of each transition in the original DataFrame
    transition_total_counts = df['transition'].value_counts().reset_index()
    transition_total_counts.columns = ['transition', 'total_count']

    return transition_pivot, transition_total_counts


def ActivityTransitionsFrequencyEncoding(df, var_id, var_activity, var_timestamp):
    """
    This function performs frequency encoding for activity transitions in a process.
    
    Given a DataFrame containing case identifiers, activities, and timestamps,
    the function identifies the transitions from one activity to another within each case
    and creates a frequency matrix of these transitions for each case.
    
    In addition,
    the function returns the total number of occurrences of each transition in the original DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - var_id (str): Name of the column containing the case identifiers (default: 'case:case').
    - var_activity (str): Name of the column containing the activities (default: 'activity').
    - timestamp_col (str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame with the frequency matrix of transitions per case.
    - pd.DataFrame: DataFrame with the total number of occurrences of each transition in the original DataFrame.
    """

    # Convert timestamp column to datetime
    df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Sort by case and timestamp
    df = df.sort_values(by=[var_id, var_timestamp])

    # Identify the next activity for each event
    df['next_activity'] = df.groupby(var_id)[var_activity].shift(-1)

    # Create the transitions column
    df['transition'] = df[var_activity] + ' -> ' + df['next_activity']

    # Remove transitions where 'next_activity' is NaN
    df = df.dropna(subset=['next_activity'])
    
    # Delete the last transition of each case
    df['is_last'] = df.groupby(var_id).cumcount(ascending=False) == 0
    df = df[~df['is_last']].copy()

    # Count transitions per case
    transition_counts = df.groupby([var_id, 'transition']).size().reset_index(name='count')

    # Pivot the data to create the transition frequency matrix
    transition_pivot = transition_counts.pivot(index=var_id, columns='transition', values='count').fillna(0)

    # Reset the index so that var_id becomes a column
    transition_pivot = transition_pivot.reset_index()

    # Count the total of each transition in the original DataFrame
    transition_total_counts = df['transition'].value_counts().reset_index()
    transition_total_counts.columns = ['transition', 'total_count']

    return transition_pivot, transition_total_counts


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Complete trace                               """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ActivityPrefixEncoding(df, var_id, var_activity, var_timestamp):
    """
    This function performs prefix encoding for activities in a process.
    Given a DataFrame containing case identifiers, activities, and timestamps,
    the function creates activity prefixes for each case by calculating the frequency
    of those prefixes across the entire dataset. The function returns a DataFrame
    with the cases, their respective prefixes, and the prefix label, as well as a
    DataFrame containing the count of each prefix and its frequency order.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - var_id (str): Name of the column containing the case identifiers (default: 'case:case').
    - var_activity (str): Name of the column containing the activities (default: 'activity').
    - var_timestamp (str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame with the cases, their prefixes and their labels.
    - pd.DataFrame: DataFrame with the prefix count and their frequency order.
    """

    # Sort the DataFrame by case IDs and timestamps
    df_sorted = df.sort_values(by=[var_id, var_timestamp])
    
    # Group activities by case and create the activity list
    df_grouped = df_sorted.groupby(var_id)[var_activity].apply(list).reset_index()
    
    # Create a new column with the prefixes (without the last activity)
    df_grouped['activity_prefix'] = df_grouped[var_activity].apply(lambda x: ' -> '.join(x[:-1]) if len(x) > 1 else '')

    # Filter only cases that have prefix
    df_grouped_with_prefix = df_grouped[df_grouped['activity_prefix'] != ''].copy()
    
    # Create the base with all possible prefixes
    prefix_data = []
    for prefix in df_grouped_with_prefix['activity_prefix']:
        activities = prefix.split(' -> ')
        prefix_data.append({'activity_prefix': prefix, 'activity_prefix_size': len(activities)})
    
    # Create the prefix DataFrame
    df_activity_prefixes = pd.DataFrame(prefix_data)
    
    # Count how many times each prefix occurs (already considering 1 prefix per case)
    df_activity_prefixes_count = df_activity_prefixes.groupby(['activity_prefix', 'activity_prefix_size']).size().reset_index(name='activity_prefix_count')
    
    # Sort by prefix frequency (most frequent to least frequent)
    df_activity_prefixes_count = df_activity_prefixes_count.sort_values(by='activity_prefix_count', ascending=False).reset_index(drop=True)
    
    # Add numeric label based on frequency order
    df_activity_prefixes_count['activity_prefix_label'] = df_activity_prefixes_count.index + 1
    
    # Associate the prefix label with df_grouped_with_prefix
    df_grouped_with_prefix = df_grouped_with_prefix.merge(df_activity_prefixes_count[['activity_prefix', 'activity_prefix_label']], on='activity_prefix', how='left')

    # Remove the original activity column
    df_grouped_with_prefix = df_grouped_with_prefix.drop(var_activity, axis=1)
    
    return df_grouped_with_prefix, df_activity_prefixes_count


def ActivityResourcePrefixEncoding(df, var_id, var_activity, var_resource, var_timestamp):
    """
    This function performs prefix encoding for the combination of activities and resources in a process.
    Given a DataFrame containing case identifiers, activities, resources, and timestamps,
    the function creates prefixes for each case by calculating the frequency of these prefixes across the entire dataset.
    The function returns a DataFrame with the cases, their respective prefixes, and the prefix label, as well as a
    DataFrame containing the count of each prefix and its frequency order.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - var_id (str): Name of the column containing the case identifiers.
    - var_activity (str): Name of the column containing the activities.
    - var_resource (str): Name of the column containing the resources.
    - var_timestamp (str): Name of the column containing the timestamps.
    
    Returns:
    - pd.DataFrame: DataFrame with the cases, their prefixes, and their labels.
    - pd.DataFrame: DataFrame with the prefix count and their frequency order.
    """

    # Sort the DataFrame by case IDs and timestamps
    df_sorted = df.sort_values(by=[var_id, var_timestamp])
    
    # Create a new column combining activity and resource
    df_sorted['activity_resource'] = df_sorted[var_activity] + ' ; ' + df_sorted[var_resource]
    
    # Group the activity and resource combinations by case and create the list of these combinations
    df_grouped = df_sorted.groupby(var_id)['activity_resource'].apply(list).reset_index()
    
    # Create a new column with the prefixes (without the last activity and resource combination)
    df_grouped['activity_resource_prefix'] = df_grouped['activity_resource'].apply(lambda x: ' -> '.join(x[:-1]) if len(x) > 1 else '')

    # Filter only cases that have prefix
    df_grouped_with_prefix = df_grouped[df_grouped['activity_resource_prefix'] != ''].copy()
    
    # Create the base with all possible prefixes
    prefix_data = []
    for prefix in df_grouped_with_prefix['activity_resource_prefix']:
        activities_resources = prefix.split(' -> ')
        prefix_data.append({'activity_resource_prefix': prefix, 'activity_resource_prefix_size': len(activities_resources)})
    
    # Create the prefix DataFrame
    df_activity_resource_prefixes = pd.DataFrame(prefix_data)
    
    # Count how many times each prefix occurs (already considering 1 prefix per case)
    df_activity_resource_prefixes_count = df_activity_resource_prefixes.groupby(['activity_resource_prefix', 'activity_resource_prefix_size']).size().reset_index(name='activity_resource_prefix_count')
    
    # Sort by prefix frequency (most frequent to least frequent)
    df_activity_resource_prefixes_count = df_activity_resource_prefixes_count.sort_values(by='activity_resource_prefix_count', ascending=False).reset_index(drop=True)
    
    # Add numeric label based on frequency order
    df_activity_resource_prefixes_count['activity_resource_prefix_label'] = df_activity_resource_prefixes_count.index + 1
    
    # Associate the prefix label with df_grouped_with_prefix
    df_grouped_with_prefix = df_grouped_with_prefix.merge(df_activity_resource_prefixes_count[['activity_resource_prefix', 'activity_resource_prefix_label']], on='activity_resource_prefix', how='left')

    # Remove the original column from activity resource
    df_grouped_with_prefix = df_grouped_with_prefix.drop('activity_resource', axis=1)
    
    return df_grouped_with_prefix, df_activity_resource_prefixes_count



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                         Activities + Resources                           """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ActivityResourceOneHotEncoding(df, var_id, var_activity, var_resource):
    """
    This function performs one-hot encoding for the combination of activities and resources in a process.
    
    Given a DataFrame containing case, activity, and resource identifiers, the function
    creates binary columns indicating the presence or absence of each activity + resource combination
    within each case. The new columns are concatenated to the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers.
    - var_activity(str): Name of the column containing the activities.
    - var_resource(str): Name of the column containing the resources.
    
    Returns:
    - pd.DataFrame: Original DataFrame concatenated with the one-hot encoding columns.
    """
    
    # Create a new column combining activity and resource
    df['activity_resource'] = df[var_activity] + ' - ' + df[var_resource]

    # Group by case_id and the new activity_resource column, checking for presence
    presence = df.groupby([var_id, 'activity_resource']).size().unstack(fill_value=0).astype(bool).astype(int)

    # Concatenate the original data with the presence encoding
    final_df = pd.merge(df, presence, on=var_id, how='left')
    
    return final_df


def ActivityResourceFrequencyEncoding(df, var_id, var_activity, var_resource):
    """
    This function performs frequency encoding for the combination of activities and resources in a process.
    
    Given a DataFrame containing case, activity, and resource identifiers, the function
    creates columns indicating the frequency of each activity + resource combination within each case.
    
    The new frequency columns are then concatenated to the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers.
    - var_activity(str): Name of the column containing the activities.
    - var_resource(str): Name of the column containing the resources.
    
    Returns:
    - pd.DataFrame: Original DataFrame concatenated with the frequency-encoded columns.
    """

    # Create a new column combining activity and resource
    df['activity_resource'] = df[var_activity] + ' - ' + df[var_resource]

    # Group by case_id and the new activity_resource column, counting frequency
    freq_counts = df.groupby([var_id, 'activity_resource']).size().unstack(fill_value=0)

    # Concatenate the original data with frequency encoding
    final_df = pd.merge(df, freq_counts, on=var_id, how='left')

    return final_df



def ActivityResourceTfIdfEncoding(df, var_id, var_activity, var_resource, var_timestamp):
    """
    This function preprocesses the log data by removing the last activity and resource combination from each trace,
    and then computes the TF-IDF representation for the remaining combinations.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the event log.
    - var_id(str): Name of the column containing the case identifiers (default: 'case:concept:name').
    - var_activity(str): Name of the column containing the activities (default: 'concept:name').
    - var_resource(str): Name of the column containing the resources (default: 'resource').
    - var_timestamp(str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame containing the TF-IDF matrix of the remaining activity and resource combinations after the last one has been removed.
    - pd.DataFrame: DataFrame with TF values (Term Frequency) for each document and term.
    - pd.DataFrame: DataFrame with IDF (Inverse Document Frequency) values ​​for each term.
    """
    
    # Preprocessing: Replace spaces with underscores in activities and resources
    df_processed = df.copy()
    df_processed[var_activity] = df_processed[var_activity].str.replace(' ', '_')
    df_processed[var_activity] = df_processed[var_activity].str.replace('-', '_')
    df_processed[var_resource] = df_processed[var_resource].str.replace(' ', '_')
    df_processed[var_resource] = df_processed[var_resource].str.replace('-', '_')
    df_processed[var_resource] = df_processed[var_resource].str.replace('.', '')

    # Create a new column combining activity and resource
    df_processed['activity_resource'] = df_processed[var_activity] + '_' + df_processed[var_resource]

    # Convert timestamp column to datetime (if applicable)
    df_processed[var_timestamp] = pd.to_datetime(df_processed[var_timestamp])

    # Sort by case ID and timestamp to ensure correct order of activities and resources
    df_sorted = df_processed.sort_values(by=[var_id, var_timestamp])

    # Mark the last activity/resource of each case (trace)
    df_sorted['is_last'] = df_sorted.groupby(var_id).cumcount(ascending=False) == 0

    # Remove the last activity/resource from each trace
    df_without_last = df_sorted[~df_sorted['is_last']].copy()

    # Concatenate all remaining activity and resource combinations from each trace as a single string
    traces = df_without_last.groupby(var_id)['activity_resource'].apply(lambda x: ' '.join(x)).reset_index()

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()

    # Adjust and transform the remaining activity and resource combinations to generate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(traces['activity_resource'])

    # Get the names of the combinations of activities and features
    feature_names = vectorizer.get_feature_names_out()

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=traces[var_id])

    # Get IDF values
    idf_values = vectorizer.idf_
    idf_df = pd.DataFrame(idf_values, index=feature_names, columns=['IDF'])

    # Calculate TF manually (term frequency divided by total number of words in each document)
    tf_matrix = vectorizer.transform(traces['activity_resource']).toarray()
    tf_df = pd.DataFrame(tf_matrix, columns=feature_names, index=traces[var_id])

    return tfidf_df, tf_df, idf_df




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""               Transition between Activities + Resources                  """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ActivityResourceTransitionsOneHotEncoding(df, var_id, var_activity, var_resource, var_timestamp):
    """
    This function performs one-hot encoding for activity and resource transitions in a process.
    
    Given a DataFrame containing case, activity, resource identifiers, and timestamps,
    the function identifies transitions from one activity and resource combination to another within each case
    and creates a presence matrix of these transitions for each case.
    
    In addition,
    the function returns the total of each transition in the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers.
    - var_activity(str): Name of the column containing the activities.
    - var_resource(str): Name of the column containing the resources.
    - var_timestamp(str): Name of the column containing the timestamps.
    
    Returns:
    - pd.DataFrame: DataFrame with the presence matrix of transitions per case.
    - pd.DataFrame: DataFrame with the total of each transition in the original DataFrame.
    """

    # Convert timestamp column to datetime
    df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Sort by case and timestamp
    df = df.sort_values(by=[var_id, var_timestamp])

    # Create a new column combining activity and resource
    df['activity_resource'] = df[var_activity] + ' - ' + df[var_resource]

    # Identify the next activity/resource for each event
    df['next_activity_resource'] = df.groupby(var_id)['activity_resource'].shift(-1)

    # Create the transitions column
    df['transition'] = df['activity_resource'] + ' -> ' + df['next_activity_resource']

    # Remove transitions where 'next_activity_resource' is NaN
    df = df.dropna(subset=['next_activity_resource'])

    # Mark the presence of transitions by case
    transition_presence = df.groupby([var_id, 'transition']).size().reset_index(name='count')
    transition_presence['count'] = 1  

    # Pivot the data to create the transition presence matrix
    transition_pivot = transition_presence.pivot(index=var_id, columns='transition', values='count').fillna(0)

    # Reset the index so that var_id becomes a column
    transition_pivot = transition_pivot.reset_index()

    # Count the total of each transition in the original DataFrame
    transition_total_counts = df['transition'].value_counts().reset_index()
    transition_total_counts.columns = ['transition', 'total_count']

    return transition_pivot, transition_total_counts


def ActivityResourceTransitionsFrequencyEncoding(df, var_id, var_activity, var_resource, var_timestamp):
    """
    This function performs frequency encoding for activity and resource transitions in a process.
    
    Given a DataFrame containing case, activity, resource identifiers, and timestamps,
    the function identifies transitions from one activity and resource combination to another within each case
    and creates a frequency matrix of these transitions for each case.
    
    In addition,
    the function returns the total number of occurrences of each transition in the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers.
    - var_activity(str): Name of the column containing the activities.
    - var_resource(str): Name of the column containing the resources.
    - var_timestamp(str): Name of the column containing the timestamps.
    
    Returns:
    - pd.DataFrame: DataFrame with the frequency matrix of transitions by case.
    - pd.DataFrame: DataFrame with the total number of occurrences of each transition in the original DataFrame.
    """

    # Convert timestamp column to datetime
    df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Sort by case and timestamp
    df = df.sort_values(by=[var_id, var_timestamp])

    # Create a new column combining activity and resource
    df['activity_resource'] = df[var_activity] + ' - ' + df[var_resource]

    # Identify the next activity/resource for each event
    df['next_activity_resource'] = df.groupby(var_id)['activity_resource'].shift(-1)

    # Create the transitions column
    df['transition'] = df['activity_resource'] + ' -> ' + df['next_activity_resource']

    # Remove transitions where 'next_activity_resource' is NaN
    df = df.dropna(subset=['next_activity_resource'])

    # Count transitions per case
    transition_counts = df.groupby([var_id, 'transition']).size().reset_index(name='count')

    # Pivotar os dados para criar a matriz de frequência de transições
    transition_pivot = transition_counts.pivot(index=var_id, columns='transition', values='count').fillna(0)

    # Reset the index so that var_id becomes a column
    transition_pivot = transition_pivot.reset_index()

    # Count the total of each transition in the original DataFrame
    transition_total_counts = df['transition'].value_counts().reset_index()
    transition_total_counts.columns = ['transition', 'total_count']

    return transition_pivot, transition_total_counts
