#!/usr/bin/python3

from utils import benchmark_imputer, sample_dataset
import numpy as np


def test_imputer():
    target_path = "./dataset.ods"

    zero_probability = 0.1

    testdata = sample_dataset(target_path, zero_probability)
    size = testdata.size
    deleted_elements_count = size - np.count_nonzero(testdata)

    (
        _,
        _,
        _,
        errors,
        mse,
    ) = benchmark_imputer(testdata, target_path)

    print("Errors:\n", errors)
    print("MSE:", mse)
    print(f"Deleted elements: {deleted_elements_count} out of {size}")
