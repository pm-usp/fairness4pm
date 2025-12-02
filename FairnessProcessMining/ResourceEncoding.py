import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Resources                                     """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ResourceOneHotEncoding(df, var_id, var_resource):
    """
    This function performs one-hot encoding for features in a case.
    
    Given a DataFrame containing case identifiers and features,
    the function creates binary columns indicating the presence or absence of each feature
    within each case. The new columns are concatenated to the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the case data.
    - var_id(str): Name of the column containing the case identifiers (default: 'case:case').
    - var_resource(str): Name of the column containing the features (default: 'resource').
    
    Returns:
    - pd.DataFrame: Original DataFrame concatenated with the one-hot encoding columns.
    """
    
    # Group by case_id and resource, checking for presence
    presence = df.groupby([var_id, var_resource]).size().unstack(fill_value=0).astype(bool).astype(int)
  
    # Concatenate the original data with presence encoding
    final_df = pd.merge(df, presence, on=var_id, how='left')

    return final_df



def ResourceFrequencyEncoding(df, var_id, var_resource):
    """
    This function performs frequency encoding for features in a case.
    
    Given a DataFrame containing case identifiers and features,
    the function creates columns indicating the frequency of each feature within each case.
    
    The new frequency columns are then concatenated to the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the case data.
    - var_id(str): Name of the column containing the case identifiers (default: 'case:case').
    - var_resource(str): Name of the column containing the features (default: 'resource').
    
    Returns:
    - pd.DataFrame: Original DataFrame concatenated with the frequency-encoded columns.
    """
    
    # Group by case_id and resource, counting frequency
    freq_counts = df.groupby([var_id, var_resource]).size().unstack(fill_value=0)
   
    # Concatenate the original data with frequency encoding
    final_df = pd.merge(df, freq_counts, on=var_id, how='left')

    return final_df



def ResourceTfIdfEncoding(df, var_id, var_resource, var_timestamp):
    """
    This function preprocesses the log data by removing the last activity (based on the resource) from each trace,
    and then calculates the TF-IDF representation for the remaining resources.
    
    TF: counts how many times an activity (term) occurs in a trace (document);
    IDF: checks in how many traces (documents) a given activity (term) occurs in the entire event log.
    TF-IDF: TF x IDF
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the event log.
    - var_id (str): Name of the column containing the case identifiers (default: 'case:concept:name').
    - var_resource (str): Name of the column containing the resources (default: 'resource').
    - var_timestamp (str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame containing the TF-IDF matrix of the remaining features after removing the last activity from each trace.
    - pd.DataFrame: DataFrame with TF (Term Frequency) values ​​for each document and term.
    - pd.DataFrame: DataFrame with IDF (Inverse Document Frequency) values ​​for each term.
    """
    
    # Preprocessing: Replace spaces with underscores in features
    df_processed = df.copy()
    df_processed[var_resource] = df_processed[var_resource].str.replace(' ', '_')
    df_processed[var_resource] = df_processed[var_resource].str.replace('-', '_')
    df_processed[var_resource] = df_processed[var_resource].str.replace('.', '')

    # Convert timestamp column to datetime (if applicable)
    df_processed[var_timestamp] = pd.to_datetime(df_processed[var_timestamp])

   # Sort by case ID and timestamp to ensure correct order of activities
    df_sorted = df_processed.sort_values(by=[var_id, var_timestamp])

    # Mark the last activity of each case (trace)
    df_sorted['is_last'] = df_sorted.groupby(var_id).cumcount(ascending=False) == 0

    # Remove the last activity from each trace
    df_without_last = df_sorted[~df_sorted['is_last']].copy()

    # Concatenate all remaining features from each trace as a single string
    traces = df_without_last.groupby(var_id)[var_resource].apply(lambda x: ' '.join(x)).reset_index()

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()

    # Adjust and transform the remaining features to generate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(traces[var_resource])

    # Get the names of the features
    feature_names = vectorizer.get_feature_names_out()

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=traces[var_id])

    # Get IDF values
    idf_values = vectorizer.idf_
    idf_df = pd.DataFrame(idf_values, index=feature_names, columns=['IDF'])

    # Calculate TF manually (term frequency divided by total number of words in each document)
    tf_matrix = vectorizer.transform(traces[var_resource]).toarray()
    tf_df = pd.DataFrame(tf_matrix, columns=feature_names, index=traces[var_id])

    return tfidf_df, tf_df, idf_df


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                    Transition between Resources                          """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ResourceTransitionsOneHotEncoding(df, var_id, var_resource, var_timestamp):
    """
    This function performs one-hot encoding for resource transitions in a process.
    
    Given a DataFrame containing case identifiers, resources, and timestamps,
    the function identifies the transitions from one resource to another within each case and
    creates a binary matrix that marks the presence of these transitions. In addition, the function
    returns the total number of occurrences of each transition in the original DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the process data.
    - var_id (str): Name of the column containing the case identifiers (default: 'case:case').
    - var_resource (str): Name of the column containing the resources (default: 'resource').
    - var_timestamp (str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame with the matrix of presence of transitions by case.
    - pd.DataFrame: DataFrame with the total number of occurrences of each transition in the original DataFrame.
    """

    # Convert timestamp column to datetime
    df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Sort by case and timestamp
    df = df.sort_values(by=[var_id, var_timestamp])

    # Identify the next resource for each event
    df['next_resource'] = df.groupby(var_id)[var_resource].shift(-1)

    # Create the transitions column
    df['transition_resource'] = df[var_resource] + ' -> ' + df['next_resource']

    # Remove transitions where 'next_resource' is NaN
    df = df.dropna(subset=['next_resource'])

    # Delete the last transition of each case
    df['is_last'] = df.groupby(var_id).cumcount(ascending=False) == 0
    df = df[~df['is_last']].copy()

    # Mark the presence of transitions by case
    transition_presence = df.groupby([var_id, 'transition_resource']).size().reset_index(name='count')
    transition_presence['count'] = 1

    # Pivot the data to create the transition presence matrix
    transition_pivot = transition_presence.pivot(index=var_id, columns='transition_resource', values='count').fillna(0)

    # Reset the index so that var_id becomes a column
    transition_pivot = transition_pivot.reset_index()

    # Count the total of each transition in the original DataFrame
    transition_total_counts = df['transition_resource'].value_counts().reset_index()
    transition_total_counts.columns = ['transition_resource', 'total_count']

    return transition_pivot, transition_total_counts


def ResourceTransitionsFrequencyEncoding(df, var_id, var_resource, var_timestamp):
    """
    This function performs frequency encoding for resource transitions in a process.
    
    Given a DataFrame containing case identifiers, resources, and timestamps,
    the function identifies the transitions from one resource to another within each case and
    creates a frequency matrix of these transitions. In addition, the function returns the total
    number of occurrences of each transition in the original DataFrame.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers (default: 'case:case').
    - var_resource(str): Name of the column containing the resources (default: 'resource').
    - var_timestamp(str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame with the frequency matrix of transitions per case.
    - pd.DataFrame: DataFrame with the total number of occurrences of each transition in the original DataFrame.
    """

    # Convert timestamp column to datetime
    df[var_timestamp] = pd.to_datetime(df[var_timestamp])

    # Sort by case and timestamp
    df = df.sort_values(by=[var_id, var_timestamp])

    # Identify the next resource for each event
    df['next_resource'] = df.groupby(var_id)[var_resource].shift(-1)

    # Create the transitions column
    df['transition_resource'] = df[var_resource] + ' -> ' + df['next_resource']

    # Remove transitions where 'next_resource' is NaN
    df = df.dropna(subset=['next_resource'])
    
    # Delete the last transition of each case
    df['is_last'] = df.groupby(var_id).cumcount(ascending=False) == 0
    df = df[~df['is_last']].copy()

    # Count transitions per case
    transition_counts = df.groupby([var_id, 'transition_resource']).size().reset_index(name='count')

    # Pivot the data to create the transition frequency matrix
    transition_pivot = transition_counts.pivot(index=var_id, columns='transition_resource', values='count').fillna(0)

    # Reset the index so that var_id becomes a column
    transition_pivot = transition_pivot.reset_index()

    # Count the total of each transition in the original DataFrame
    transition_total_counts = df['transition_resource'].value_counts().reset_index()
    transition_total_counts.columns = ['transition_resource', 'total_count']

    return transition_pivot, transition_total_counts



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Complete trace                               """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def ResourcePrefixEncoding(df, var_id, var_resource, var_timestamp):
    """
    This function performs prefix encoding for resources in a process.
    Given a DataFrame containing case identifiers, resources, and timestamps,
    the function creates resource prefixes for each case by calculating the frequency
    of those prefixes across the entire dataset. The function returns a DataFrame
    with the cases, their respective prefixes, and the prefix label, as well as a
    DataFrame containing the count of each prefix and its frequency order.
    
    Parameters:
    - df(pd.DataFrame): DataFrame containing the process data.
    - var_id(str): Name of the column containing the case identifiers (default: 'case:case').
    - var_resource(str): Name of the column containing the resources (default: 'resource').
    - var_timestamp(str): Name of the column containing the timestamps (default: 'time:timestamp').
    
    Returns:
    - pd.DataFrame: DataFrame with the cases, their prefixes and their labels.
    - pd.DataFrame: DataFrame with the prefix count and their frequency order.
    """

    # Sort the DataFrame by case IDs and timestamps
    df_sorted = df.sort_values(by=[var_id, var_timestamp])
    
    # Group resources by case and create resource list
    df_grouped = df_sorted.groupby(var_id)[var_resource].apply(list).reset_index()
    
    # Create a new column with the prefixes (without the last resort)
    df_grouped['resource_prefix'] = df_grouped[var_resource].apply(lambda x: ' -> '.join(x[:-1]) if len(x) > 1 else '')

    # Filter only cases that have prefix
    df_grouped_with_prefix = df_grouped[df_grouped['resource_prefix'] != ''].copy()
    
    # Create the base with all possible prefixes
    prefix_data = []
    for prefix in df_grouped_with_prefix['resource_prefix']:
        resources = prefix.split(' -> ')
        prefix_data.append({'resource_prefix': prefix, 'resource_prefix_size': len(resources)})
    
    # Create the prefix DataFrame
    df_resource_prefixes = pd.DataFrame(prefix_data)
    
    # Count how many times each prefix occurs (already considering 1 prefix per case)
    df_resource_prefixes_count = df_resource_prefixes.groupby(['resource_prefix', 'resource_prefix_size']).size().reset_index(name='resource_prefix_count')
    
    # Sort by prefix frequency (most frequent to least frequent)
    df_resource_prefixes_count = df_resource_prefixes_count.sort_values(by='resource_prefix_count', ascending=False).reset_index(drop=True)
    
    # Add numeric label based on frequency order
    df_resource_prefixes_count['resource_prefix_label'] = df_resource_prefixes_count.index + 1
    
    # Associate the prefix label with df_grouped_with_prefix
    df_grouped_with_prefix = df_grouped_with_prefix.merge(df_resource_prefixes_count[['resource_prefix', 'resource_prefix_label']], on='resource_prefix', how='left')

    # Remove the original resource column
    df_grouped_with_prefix = df_grouped_with_prefix.drop(var_resource, axis=1)
    
    return df_grouped_with_prefix, df_resource_prefixes_count




