"""
stat_test.py

This module provides functions for performing various statistical tests and analyses on
multivariate data sets. It includes methods for checking normality, computing the Hotelling
T-squared statistic with PCA, conducting permutation tests, performing the Mann-Whitney U
test, and executing bootstrap resampling and comparison. The module also includes
visualization of bootstrap results using KDE plots.
"""

import numpy as np
import pandas as pd
from scipy.stats import shapiro, f, mannwhitneyu
from sklearn.decomposition import PCA


def check_normality(df: pd.DataFrame, p_value_threshold: float = 0.05) -> tuple[bool, float]:
    """
    Check univariate normality of each feature in the DataFrame.

    Parameters
    ----------
    df : array_like
        Array of sample data.

    p_value_threshold : float
        default = 0.05

    Returns
    -------
    normal : bool
        True if x has a normal distribution (p-value >= p_value_threshold)
    p-value : float
        The p-value for the hypothesis test.
    """
    for column in df.columns:
        _, p = shapiro(df[column])
        if p < p_value_threshold:
            return False, p
    return True, p


def hotelling_t2_with_pca(df1, df2, n_components=None, alpha=0.05):
    """
    Compute the Hotelling T-squared statistic for comparing two samples after reducing
    dimensionality using PCA.

    Parameters:
    df1 : pandas DataFrame
        The first sample data.
    df2 : pandas DataFrame
        The second sample data.
    n_components : int, optional
        Number of principal components to retain.
    alpha : float, optional
        Significance level for the test.

    Returns:
    T2 : float
        The Hotelling T-squared statistic.
    p_value : float
        The p-value associated with the test.
    different : bool
        Whether the samples are significantly different.
    """

    # Check for normality
    if not check_normality(df1)[0] or not check_normality(df2)[0]:
        print("Data is not normally distributed. Consider transforming the data or using"
              " a non-parametric test.")
        return None, None, None

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca.fit(pd.concat([df1, df2], axis=0))

    # Convert dataframes to numpy arrays
    sample1 = pd.DataFrame(pca.transform(df1)).values
    sample2 = pd.DataFrame(pca.transform(df2)).values

    # Number of samples in each group
    n1, p1 = sample1.shape

    if p1 != sample2.shape[1]:
        raise ValueError("Both samples must have the same number of features after PCA.")

    # Pooled covariance matrix
    sp = ((n1 - 1) * np.cov(sample1, rowvar=False)
          + (sample2.shape[0] - 1) * np.cov(sample2, rowvar=False)) / (n1 + sample2.shape[0] - 2)

    # Difference between mean vectors
    mean_diff = np.mean(sample1, axis=0) - np.mean(sample2, axis=0)

    # Compute Hotelling's T-squared statistic
    t2 = ((n1 * sample2.shape[0]) / (n1 + sample2.shape[0]) * mean_diff.T @ np.linalg.pinv(sp)
          @ mean_diff)

    # Degrees of freedom
    d_f2 = n1 + sample2.shape[0] - p1 - 1

    # Ensure valid degrees of freedom
    if d_f2 <= 0:
        raise ValueError(
            f"Invalid degrees of freedom (df2={d_f2}). Ensure sample sizes and number of features"
            f" are sufficient.")

    # Compute the F-statistic
    f_stat = (d_f2 / (p1 * (n1 + sample2.shape[0] - 2))) * t2

    # Compute the p-value
    p_value = 1 - f.cdf(f_stat, p1, d_f2)

    return t2, p_value, p_value < alpha


def permutation_test(df1, df2, num_permutations=10000, alpha=0.05):
    """
    Perform a permutation test to compare two multivariate samples.

    Parameters:
    df1 : pandas DataFrame
        The first sample data.
    df2 : pandas DataFrame
        The second sample data.
    num_permutations : int, optional
        Number of permutations to perform.
    alpha : float, optional
        Significance level for the test.

    Returns:
    test_stat : float
        The observed test statistic.
    p_value : float
        The p-value associated with the test.
    different : bool
        Whether the samples are significantly different.
    """

    # Combine the data
    combined_data = pd.concat([df1, df2], axis=0)
    n1 = len(df1)

    # Compute the observed test statistic (Euclidean distance between means)
    observed_stat = np.linalg.norm(df1.mean().values - df2.mean().values)

    # Permutation test
    count = 0
    for _ in range(num_permutations):
        permuted_data = combined_data.sample(frac=1).reset_index(drop=True)
        permuted_df1 = permuted_data.iloc[:n1]
        permuted_df2 = permuted_data.iloc[n1:]
        perm_stat = np.linalg.norm(permuted_df1.mean().values - permuted_df2.mean().values)
        if perm_stat >= observed_stat:
            count += 1

    p_value = count / num_permutations

    return observed_stat, p_value, p_value < alpha


def mann_whitney_u_test(df1, df2, alpha=0.05):
    """
    Perform Mann-Whitney U test on each feature of two multivariate samples.

    Parameters
    -------
    df1 : pandas DataFrame
        The first sample data.
    df2 : pandas DataFrame
        The second sample data.
    alpha : float, optional
        Significance level for the test.

    Returns
    -------
    results : pandas DataFrame
        A DataFrame containing test statistics and p-values for each feature.
    different : bool
        Whether there are significant differences in any feature.
    """

    # Prepare the results dataframe
    features = df1.columns
    results = pd.DataFrame(index=features, columns=['U Statistic', 'P-value'])

    for feature in features:
        data1 = df1[feature]
        data2 = df2[feature]

        # Perform the Mann-Whitney U test
        u, p_value = mannwhitneyu(data1, data2)

        # Store the results
        results.loc[feature] = [u, p_value]

    # Determine if any p-value is less than alpha
    different = results['P-value'].astype(float).min() < alpha

    return results, different
