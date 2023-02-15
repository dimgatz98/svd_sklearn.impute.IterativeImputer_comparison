#!/usr/bin/python3

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from SVD import SVDRecommender
import numpy as np
from numpy import linalg as la


def compute_squared_errors(mask, target, res):
    error = np.square(target - res)
    masked_error = np.ma.MaskedArray(error, mask)
    return masked_error


def print_distance(test_set, target, res, verbose=False):
    if np.count_nonzero(np.isnan(test_set).astype(int)) > 0:
        mask = ~np.isnan(test_set).astype(bool)
    else:
        mask = ~(test_set == 0).astype(bool)

    errors = compute_squared_errors(mask, target, res)
    mse = np.mean(errors)
    if verbose:
        print("errors:\n", errors)
        print("\nmean_error:\n", mse)
    return errors, mse


def benchmark_imputer(traindata, testdata, target, iter=10, verbose=False):
    if verbose:
        print(
            "traindata:\n", traindata, "\ntestdata:\n", testdata, "\ntarget:\n", target
        )

    testdata_df = pd.DataFrame(testdata)
    masked_testdata = np.ma.MaskedArray(testdata_df, testdata_df == 0)
    non_zero_testdata = masked_testdata.filled(np.nan)

    traindata_df = pd.DataFrame(traindata)
    masked_traindata = np.ma.MaskedArray(traindata_df, traindata_df == 0)
    non_zero_traindata = masked_traindata.filled(np.nan)

    recommender = IterativeImputer(max_iter=iter, random_state=0)
    recommender.fit(non_zero_traindata)
    res = recommender.transform(non_zero_testdata)

    errors, mse = print_distance(testdata, target, res, verbose)

    return res, target, non_zero_testdata, errors, mse


def benchmark_svd(testdata, target, verbose=False):
    if verbose:
        print("\ntestdata:\n", testdata, "\ntarget:\n", target)

    testdata_df = pd.DataFrame(testdata)
    masked_testdata = np.ma.MaskedArray(testdata_df, testdata_df == 0)
    non_zero_testdata = masked_testdata.filled(np.nan)

    recommender = SVDRecommender()
    res = recommender.fill_matrix(testdata)

    errors, mse = print_distance(testdata, target, res, verbose)

    return res, target, non_zero_testdata, errors, mse


def sample_dataset(target_arr, zero_probability):
    nums = (
        np.random.rand(len(target_arr), len(target_arr[0])) > zero_probability
    ).astype(int)
    res = target_arr * nums
    return res
