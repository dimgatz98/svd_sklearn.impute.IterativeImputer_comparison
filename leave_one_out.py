#!/usr/bin/python3

import time
import pandas as pd
from utils import sample_dataset, benchmark_svd, benchmark_imputer
import numpy as np
import os
import sys
import random


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


target_path = "./dataset.ods"

if __name__ == "__main__":
    target = pd.read_csv(target_path, sep="\t")
    target = target.set_index("index")
    target_arr = target.to_numpy()
    size = target_arr.size
    dirname, _ = os.path.split(os.path.abspath(__file__))
    repeat = 10
    iter = 10
    verbose = False

    res_dir = f"{dirname}/results/leave_one_out"

    columns = [
        "selected_row",
        "deleted_elements_count",
        "svd_mse",
        "svd_delay",
        "imputer_mse",
        "imputer_delay",
    ]
    if os.path.isfile(res_dir):
        print(f"{res_dir} is a file")
        sys.exit(0)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    df = pd.DataFrame([], columns=columns)
    for row in range(18):
        selected_row = target.index[row]

        for deleted_elements_count in range(1, 6):
            if deleted_elements_count == 2:
                verbose = True
            else:
                verbose = False
            df.loc[len(df)] = [None] * len(columns)
            for _ in range(repeat):

                zero_elements = random_combination(
                    [0, 1, 2, 3, 4, 5], deleted_elements_count
                )

                reduced_data = target_arr.copy()
                for el in zero_elements:
                    reduced_data[row][el] = 0

                try:
                    start = time.time()
                    (
                        svd_res,
                        svd_target,
                        svd_non_zero_testdata,
                        svd_errors,
                        svd_mse,
                    ) = benchmark_svd(
                        reduced_data,
                        target,
                        verbose=verbose,
                    )
                    svd_delay = time.time() - start

                    start = time.time()
                    (
                        imputer_res,
                        imputer_target,
                        imputer_non_zero_testdata,
                        imputer_errors,
                        imputer_mse,
                    ) = benchmark_imputer(
                        reduced_data,
                        reduced_data,
                        target,
                        iter=iter,
                        verbose=verbose,
                    )
                    imputer_delay = time.time() - start
                except:
                    df = df.drop(len(df) - 1)
                    continue

                for col in columns:
                    if pd.isnull(df.loc[len(df) - 1, col]):
                        df.loc[len(df) - 1, col] = (
                            eval(col)
                            if col in ["deleted_elements_count", "selected_row"]
                            else np.round(eval(col) / repeat, 3)
                        )
                    else:
                        if col in ["deleted_elements_count", "selected_row"]:
                            continue
                        else:
                            df.loc[len(df) - 1, col] += np.round(eval(col) / repeat, 3)

    df = df.sort_values(by=["deleted_elements_count"])
    df.to_csv(f"{res_dir}/results.ods", mode="w", header=True, index=False, sep="\t")
