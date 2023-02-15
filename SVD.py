#!/usr/bin/python3

import numpy as np

from numpy import linalg as la
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def get_k(sigma, percentage):
    sigma_sqr = sigma**2
    sum_sigma_sqr = sum(sigma_sqr)
    k_sum_sigma = 0
    k = 0
    for i in sigma:
        k_sum_sigma += i**2
        k += 1
        if k_sum_sigma >= sum_sigma_sqr * percentage:
            return k


def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


class SVDRecommender:
    def __init__(self):
        self.factors = None

    def svdEst(self, testdata, user, simMeas, item, percentage):
        if self.factors is None:
            u, sigma, vt = la.svd(testdata)
            self.factors = (u, sigma, vt)
        else:
            u, sigma, vt = self.factors
        n = np.shape(testdata)[1]
        sim_total = 0.0
        rat_sim_total = 0.0
        k = get_k(sigma, percentage)

        # Construct the diagonal matrix
        sigma_k = np.diag(sigma[:k])

        # Convert the original data to k-dimensional space (lower dimension) according to the value of k. formed_items represents the value of item in k-dimensional space after conversion.
        formed_items = np.around(
            np.dot(np.dot(u[:, :k], sigma_k), vt[:k, :]), decimals=3
        )
        for j in range(n):
            user_rating = testdata[user, j]
            if user_rating == 0 or j == item:
                continue
            # the similarity between item and item j
            similarity = simMeas(formed_items[item, :].T, formed_items[j, :].T)
            sim_total += similarity

            # product of similarity and the rating of user to item j, then sum
            rat_sim_total += similarity * user_rating
        if sim_total == 0:
            return 0
        else:
            return np.round(rat_sim_total / sim_total, decimals=3)

    def recommend(self, testdata, user, sim_meas, est_method, percentage=0.9):
        unrated_items = np.nonzero(testdata[user, :] == 0)[0].tolist()

        if len(unrated_items) == 0:
            return None
        res = testdata
        item_scores = []
        for item in unrated_items:
            estimated_score = est_method(testdata, user, sim_meas, item, percentage)
            res[user, item] = estimated_score
            item_scores.append((item, estimated_score))
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
        return item_scores

    def fill_matrix(self, testdata):
        testdata = testdata.copy()
        users = len(testdata)
        recommendations = []

        for user in range(users):
            recommendation = self.recommend(
                testdata=testdata,
                user=user,
                sim_meas=ecludSim,
                est_method=self.svdEst,
                percentage=0.9,
            )
            if recommendation is None:
                recommendations.append(None)
            else:
                recommendations.append(recommendation)

        res = testdata
        for user, r in enumerate(recommendations):
            if r is None:
                continue
            for el in r:
                res[user, el[0]] = el[1]

        return res
