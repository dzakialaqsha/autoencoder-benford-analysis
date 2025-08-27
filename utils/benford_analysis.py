import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_benford(dataset, first_digit_column):
    '''
    This functions generate a visualization on the conformity to Benford's distribution of first digits.

    Args:
      dataset (pd.DataFrame) = The full dataset to be analyzed.
      first_digit_column (str) = The name of the numerical column of which the first digits will be analyzed.

    Returns:
      None
    '''
    if not isinstance(dataset, pd.DataFrame):
      raise ValueError("Error. The 'data' argument must be a pandas DataFrame.")
    
    if not isinstance(first_digit_column, str):
      raise ValueError("Error. The 'first_digit_column' argument must be a string.")

    if first_digit_column not in dataset.columns:
      raise ValueError(f"Error. The 'first_digit_column' argument must be a column in the provided DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(dataset[first_digit_column].dtype):
      raise ValueError(f"Error. The 'first_digit_column: {first_digit_column}' argument must be a numeric column in the provided DataFrame.")

    #Identify first digit
    dataset = dataset.copy()
    dataset['first_digit'] = dataset[first_digit_column].astype(str).str[0].astype(int)

    #calculate observed counts
    observed_counts = dataset.first_digit.value_counts(normalize=False).sort_index()
    total_observations = dataset.first_digit.count()

    all_digits = pd.Series(0, index=range(1, 10))
    observed_counts = all_digits.add(observed_counts, fill_value=0)

    benford_probabilities = {d: np.log10(1 + 1/d) for d in range(1, 10)}
    benford_expected_counts = {d: prob * total_observations for d, prob in benford_probabilities.items()}
    benford_expected_counts_series = pd.Series(benford_expected_counts)
    observed_values = observed_counts.loc[range(1, 10)].values
    expected_values = benford_expected_counts_series.loc[range(1, 10)].values

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(dataset.first_digit, bins=np.arange(0.5, 10.5, 1), kde=False, stat="count", color='skyblue', label='Observed Frequencies')

    plt.plot(benford_expected_counts_series.index, benford_expected_counts_series.values,
             color='red', linestyle='--', marker='o', label="Benford's Law Expected Counts")

    plt.title("Distribution of First Digits vs. Benford's Law")
    plt.xlabel("First Digit")
    plt.ylabel("Count")
    plt.xticks(range(1, 10))
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    plt.show()

def test_benford(dataset, first_digit_column):
    '''
    This functions aims to calculate the MAD conformity score of the first digits of the numerical column named first_digit_column to Benford's distribution of first digits

    Args:
      dataset (pd.Dataframe): The full dataset to be analyzed.
      first_digit_column (str): The name of the numerical column of which the first digits will be analyzed.

    Returns:
      mad (np.float64): The Mean Absolute Deviation of the first digits of the column.
      conformity (str): The conformity category of the first digits of the column.
    '''
    if not isinstance(dataset, pd.DataFrame):
      raise ValueError("Error. The 'data' argument must be a pandas DataFrame.")
    
    if not isinstance(first_digit_column, str):
      raise ValueError("Error. The 'first_digit_column' argument must be a string.")

    if first_digit_column not in dataset.columns:
      raise ValueError(f"Error. The 'first_digit_column' argument must be a column in the provided DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(dataset[first_digit_column].dtype):
      raise ValueError(f"Error. The 'first_digit_column: {first_digit_column}' argument must be a numeric column in the provided DataFrame.")

    dataset = dataset.copy()
    
    #Identify first digit
    dataset = dataset.copy()
    dataset['first_digit'] = dataset[first_digit_column].astype(str).str[0].astype(int)

    #calculate observed counts
    observed_counts = dataset.first_digit.value_counts(normalize=False).sort_index()
    total_observations = dataset.first_digit.count()

    all_digits = pd.Series(0, index=range(1, 10))
    observed_counts = all_digits.add(observed_counts, fill_value=0)

    benford_probabilities = {d: np.log10(1 + 1/d) for d in range(1, 10)}
    benford_probabilities_series = pd.Series(benford_probabilities)
    benford_expected_counts = {d: prob * total_observations for d, prob in benford_probabilities.items()}
    benford_expected_counts_series = pd.Series(benford_expected_counts)
    observed_values = observed_counts.loc[range(1, 10)].values
    expected_values = benford_expected_counts_series.loc[range(1, 10)].values
    observed_proportions = observed_counts / total_observations
    mad = np.mean(np.abs(observed_proportions.loc[range(1, 10)].values - benford_probabilities_series.loc[range(1, 10)].values))

    conformity = ''
    #define conformity category
    if mad < 0.004:
      conformity = 'Close Conformity'
    elif mad >= 0.004 and mad < 0.008:
      conformity = 'Acceptable Conformity'
    elif mad >= 0.008 and mad < 0.012:
      conformity = 'Acceptable Deviation'
    else:
      conformity = 'Significant Deviation'

    return mad, conformity

def classwise_benford(dataset, first_digit_column, target_column):
    '''
    Analyzes the conformity of first digits to Benford's Law for each unique category in a specified column.

    This function iterates through each unique value in the `target_column` and applies a Benford's Law
    test to the corresponding subset of the dataset. It then prints the Mean Absolute Deviation (MAD)
    score and a descriptive conformity category for each group.

    Args:
        dataset (pd.DataFrame): The full dataset to be analyzed.
        first_digit_column (str): The name of the numerical column whose first digits will be analyzed.
        target_column (str): The name of the categorical column used to group the analysis.

    Returns:
        None: This function prints the results directly and does not return any value.
    '''
    if not isinstance(dataset, pd.DataFrame):
      raise ValueError("Error. The 'data' argument must be a pandas DataFrame.")

    if not isinstance(first_digit_column, str):
      raise ValueError("Error. The 'first_digit_column' argument must be a string.")

    if not isinstance(target_column, str):
      raise ValueError("Error. The 'target_column' argument must be a string.")

    if first_digit_column not in dataset.columns:
      raise ValueError(f"Error. Column {first_digit_column} doesn't exist in the provided dataset.")

    if target_column not in dataset.columns:
      raise ValueError(f"Error. Column {target_column} doesn't exist in the provided dataset.")

    if not pd.api.types.is_numeric_dtype(dataset[first_digit_column].dtype):
      raise ValueError(f"Error. The 'first_digit_column: {first_digit_column}' argument must be a numeric column in the provided DataFrame.")

    dataset = dataset.copy()

    for cat in dataset[target_column].unique().tolist():
      mad, type = test_benford(dataset[dataset[target_column]==cat], first_digit_column)
      if mad < 0.004:
        type = 'Close Conformity'
      elif mad >= 0.004 and mad < 0.008:
        type = 'Acceptable Conformity'
      elif mad >= 0.008 and mad < 0.012:
        type = 'Acceptable Deviation'
      else:
        type = 'Significant Deviation'
      print(f'{target_column} : {cat} | Mean absolute Deviation : {mad:.4f} | type = {type}')

def classwise_benford_table(dataset, first_digit_column, target_column):
    '''
    Analyzes the conformity of first digits to Benford's Law for each unique category in a specified column.

    This function iterates through each unique value in the `target_column` and applies a Benford's Law
    test to the corresponding subset of the dataset. It then prints the Mean Absolute Deviation (MAD)
    score and a descriptive conformity category for each group.

    Args:
        dataset (pd.DataFrame): The full dataset to be analyzed.
        first_digit_column (str): The name of the numerical column whose first digits will be analyzed.
        target_column (str): The name of the categorical column used to group the analysis.

    Returns:
        dataset (pd.DataFrame): The dataset with added 'benford_conformity' and 'benford_mad' columns.
    '''
    if not isinstance(dataset, pd.DataFrame):
      raise ValueError("Error. The 'data' argument must be a pandas DataFrame.")

    if not isinstance(first_digit_column, str):
      raise ValueError("Error. The 'first_digit_column' argument must be a string.")

    if not isinstance(target_column, str):
      raise ValueError("Error. The 'target_column' argument must be a string.")

    if first_digit_column not in dataset.columns:
      raise ValueError(f"Error. Column {first_digit_column} doesn't exist in the provided dataset.")

    if target_column not in dataset.columns:
      raise ValueError(f"Error. Column {target_column} doesn't exist in the provided DataFrame.")

    if not pd.api.types.is_numeric_dtype(dataset[first_digit_column].dtype):
      raise ValueError(f"Error. The 'first_digit_column: {first_digit_column}' argument must be a numeric column in the provided DataFrame.")

    dataset = dataset.copy()

    dataset["benford_conformity"] = ""
    dataset["benford_mad"] = 0.0

    for cat in dataset[target_column].unique().tolist():
        mask = (dataset[target_column] == cat)
        mad, conformity = test_benford(dataset[mask].copy(), first_digit_column)

        # Assign conformity and MAD to the filtered rows using .loc
        dataset.loc[mask, "benford_conformity"] = conformity
        dataset.loc[mask, "benford_mad"] = mad


    return dataset
