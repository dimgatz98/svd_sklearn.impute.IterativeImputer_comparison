# svd_sklearn.impute.IterativeImputer_comparison

Compare the MSE (Mean Squared Error) when trying to predict missing values from a small dataset using the techniques of collaborative filtering with SVD (Singular Value Decomposition) and sklearn.impute.IterativeImputer. For the train/test split we use two cases: 
1. Leave-One-Out Cross-Validation (LOOCV)
2. Random Sampling.

In both cases we start by deleting only a small fraction of the rated items (columns) and slowly increase this amount. 

Results can be found in results/{leane_one_out,random_sampling}.

```sh
# For leave one out
python3 leave_one_out.py
# For random sampling
python3 random_sampling.py
```