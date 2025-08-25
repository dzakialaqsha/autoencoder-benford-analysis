import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_preprocessing(data, categorical_column_names, numeric_column_names):
    '''
    Performs one-hot encoding and Min-Max scaling on specified columns of a pandas DataFrame.

    This function first validates the input DataFrame and column lists to ensure
    they exist and are of the correct type. If validation is successful, it
    applies one-hot encoding to the categorical columns and Min-Max scaling to
    the numerical columns.

    Args:
        data (pd.DataFrame): The input DataFrame to be preprocessed.
        categorical_column_names (list): A list of column names to be one-hot encoded.
        numeric_column_names (list): A list of column names to be Min-Max scaled.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with transformed columns.

    Raises:
        ValueError: If any of the input validation checks fail (e.g., incorrect type,
                    missing columns, or non-numeric data in numeric columns).
    '''
    #INPUT VALIDATION
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Error. The 'data' argument must be a pandas DataFrame.")

    if not isinstance(categorical_column_names, list):
        raise ValueError("Error. The 'categorical_column_names' argument must be a list.")

    if not isinstance(numeric_column_names, list):
        raise ValueError("Error. The 'numeric_column_names' argument must be a list.")

    if not categorical_column_names and not numeric_column_names:
        raise ValueError("Error. At least one of 'categorical_column_names' or 'numeric_column_names' must be provided.")

    all_provided_columns = categorical_column_names + numeric_column_names

    for col in all_provided_columns:
        if col not in data.columns:
            raise ValueError(f"Error. Column '{col}' does not exist in the DataFrame.")

    non_numeric_cols = [col for col in numeric_column_names if not pd.api.types.is_numeric_dtype(data[col].dtype)]

    if non_numeric_cols:
        raise ValueError(f"Error. The following columns provided as numerical are not numerical:\n{non_numeric_cols}")

    data = data.copy()
    #PREPROCESSING
    #1. One hot encoding for categorical variables
    data = pd.get_dummies(data, columns=categorical_column_names)

    #2. Min Max Scaling for numerical variables
    for col in numeric_column_names:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    for col in data.columns.tolist():
      data[col] = data[col].astype(float)

    data = data.to_numpy()

    return data

def benford_distribution_preserving_sampling(data, categorical_column_name, numeric_column_name, total_sample_size):
  '''
    Performs stratified sampling to preserve the first-digit Benford distribution within a dataset.

    This function samples data in a way that the first-digit distribution of the specified
    numeric column is maintained in the final sample, both overall and within each
    category of the specified categorical column. This is useful for creating
    subsets of data that retain a key statistical property of the original dataset.

    Args:
        data (pd.DataFrame): The input DataFrame from which to sample.
        categorical_column_name (str): The name of the categorical column used for stratification.
        numeric_column_name (str): The name of the numeric column whose first-digit
                                   distribution will be preserved.
        total_sample_size (int): The desired total number of rows in the final sample.

    Returns:
        pd.DataFrame: A new DataFrame containing the sampled data, with its first-digit
                      distribution matching the original data's distribution.

    Raises:
        ValueError: If any of the input validation checks fail, such as incorrect data types,
                    non-existent columns, or an invalid sample size.
  '''
  if not isinstance(data, pd.DataFrame):
    raise ValueError("Error. Provided data argument is not a Pandas DataFrame object")

  if not isinstance(categorical_column_name, str):
    raise ValueError("Error. Provided categorical_column_name argument is not a string")

  if not isinstance(total_sample_size, int):
    raise ValueError("Error. Provided total_sample_size argument is not an integer")

  if not categorical_column_name in data.columns:
    raise ValueError("Error. Provided categorical column do not exist in the provided data")

  if total_sample_size <= 0:
    raise ValueError("Error. Provided total_sample_size argument is not a positive integer")

  if total_sample_size > len(data):
    raise ValueError("Error. Provided total_sample_size argument is greater than the number of rows in the provided data")

  if not numeric_column_name in data.columns:
    raise ValueError("Error. Provided numeric column do not exist in the provided data")

  if not isinstance(numeric_column_name, str):
    raise ValueError("Error. Provided numeric_column_name argument is not a string")

  # Create a copy to prevent modifying the original DataFrame
  data_copy = data.copy()

  data_copy['first_digit'] = data_copy[numeric_column_name].astype(str).str[0].astype(int)

  #Get account proportion size
  account_proportion = data_copy.groupby([categorical_column_name, 'first_digit']).size().reset_index()
  account_proportion.columns = [categorical_column_name, 'first_digit', 'intra_account_digit_count']
  account_proportion['final_sample_proportion'] = account_proportion.intra_account_digit_count / len(data_copy)
  account_proportion['final_sample_size'] = (account_proportion.final_sample_proportion * total_sample_size).astype(int)
  account_proportion['final_sample_size'] = account_proportion['final_sample_size'].apply(lambda x: 1 if x == 0 else x)

  sampled_data_list = []

  for index, row in account_proportion.iterrows():
    sampled_data_list.append(data_copy[(data_copy[categorical_column_name] == row[categorical_column_name]) & (data_copy['first_digit'] == row['first_digit'])].sample(n=row['final_sample_size'], replace=False, random_state=42))

  sampled_data = pd.concat(sampled_data_list, ignore_index=True)
  sampled_data = sampled_data.drop(['first_digit'], axis='columns')

  return sampled_data
