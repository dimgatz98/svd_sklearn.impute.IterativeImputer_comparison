#!/usr/bin/python3

import time
import pandas as pd
from utils import sample_dataset, benchmark_svd, benchmark_imputer
import numpy as np
import os
import sys

target_path = "./dataset.ods"

if __name__ == "__main__":
    target = pd.read_csv(target_path, sep="\t")
    target = target.set_index("index")
    target_arr = target.to_numpy()
    size = target_arr.size
    dirname, _ = os.path.split(os.path.abspath(__file__))

    res_dir = f"{dirname}/results/random_sampling"

    columns = [
        "deleted_elements_count",
        "svd_mean_error",
        "svd_delay",
        "imputer_mean_error",
        "imputer_delay",
    ]
    if os.path.isfile(res_dir):
        print(f"{res_dir} is a file")
        sys.exit(0)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for iter in range(10, 100, 10):
        df = pd.DataFrame([], columns=columns)
        for prob in np.arange(0.1, 0.7, 0.05):
            df.loc[len(df)] = [None] * len(columns)
            for _ in range(100):
                testdata = sample_dataset(target_arr, prob)
                deleted_elements_count = size - np.count_nonzero(testdata)

                try:
                    start = time.time()
                    (
                        svd_res,
                        svd_target,
                        svd_non_zero_testdata,
                        svd_errors,
                        svd_mean_error,
                    ) = benchmark_svd(testdata, target_path)
                    svd_delay = time.time() - start

                    start = time.time()
                    (
                        imputer_res,
                        imputer_target,
                        imputer_non_zero_testdata,
                        imputer_errors,
                        imputer_mean_error,
                    ) = benchmark_imputer(testdata, target_path, iter=iter)
                    imputer_delay = time.time() - start
                except:
                    df = df.drop(len(df) - 1)
                    continue

                for col in columns:
                    if pd.isnull(df.loc[len(df) - 1, col]):
                        df.loc[len(df) - 1, col] = (
                            f"{np.round(eval(col) / 100, 3)} / {size}"
                            if col == "deleted_elements_count"
                            else np.round(eval(col), 3) / 100
                        )
                    else:
                        if col == "deleted_elements_count":
                            df.loc[len(df) - 1, col] = np.round(
                                float(df.loc[len(df) - 1, col].split("/")[0].strip(" "))
                                + eval(col) / 100,
                                3,
                            )
                            df.loc[
                                len(df) - 1, col
                            ] = f"{df.loc[len(df) - 1, col]} / {size}"
                        else:
                            df.loc[len(df) - 1, col] += np.round(eval(col) / 100, 3)

        df.to_csv(f"{res_dir}/{iter}.ods", mode="w", header=True, index=False, sep="\t")
