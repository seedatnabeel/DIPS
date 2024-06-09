"""
Adapted from https://github.com/Kthyeon/FINE_official
https://github.com/jaychoi12/FINE

FINE Samples for Learning with Noisy Labels
"""

import torch
import numpy as np
import pandas as pd
from sklearn import cluster
from tqdm import tqdm

import numpy as np
import math
import scipy.stats as stats
import torch

from sklearn.mixture import GaussianMixture as GMM


def fit_mixture(scores, labels, p_threshold=0.5):
    """
    Assume the distribution of scores: bimodal gaussian mixture model

    return clean labels
    that belongs to the clean cluster by fitting the score distribution to GMM
    """

    clean_labels = []
    indexes = np.array(range(len(scores)))
    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        feats = scores[labels == cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        gmm = GMM(n_components=2, covariance_type="full", tol=1e-6, max_iter=100)

        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:, gmm.means_.argmax()]
        clean_labels = prob > p_threshold
        clean_labels = np.where(clean_labels == 1)[0]

    return clean_labels


def get_mean_vector(features, labels):
    mean_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            v = np.mean(features[labels == index], axis=0)
            mean_vector_dict[index] = v
            pbar.update(1)

    return mean_vector_dict


def get_singular_vector(features, labels):
    """
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    """

    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels == index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict


def get_score(singular_vector_dict, features, labels, normalization=True):
    """
    Calculate the score providing the degree of showing whether the data is clean or not.
    """
    if normalization:
        scores = [
            np.abs(
                np.inner(
                    singular_vector_dict[labels[indx]], feat / np.linalg.norm(feat)
                )
            )
            for indx, feat in enumerate(tqdm(features))
        ]
    else:
        scores = [
            np.abs(np.inner(singular_vector_dict[labels[indx]], feat))
            for indx, feat in enumerate(tqdm(features))
        ]

    return np.array(scores)


def extract_topk(scores, labels, k):
    """
    k: ratio to extract topk scores in class-wise manner
    To obtain the most prominsing clean data in each classes

    return selected labels
    which contains top k data
    """

    indexes = torch.tensor(range(len(labels)))
    selected_labels = []
    for cls in np.unique(labels):
        num = int(p * np.sum(labels == cls))
        _, sorted_idx = torch.sort(scores[labels == cls], descending=True)
        selected_labels += indexes[labels == cls][sorted_idx[:num]].numpy().tolist()

    return torch.tensor(selected_labels, dtype=torch.int64)


def fine(
    current_features,
    current_labels,
    fit="kmeans",
    prev_features=None,
    prev_labels=None,
    p_threshold=0.5,
    norm=True,
    eigen=True,
):
    """
    prev_features, prev_labels: data from the previous round
    current_features, current_labels: current round's data

    return clean labels

    if you insert the prev_features and prev_labels to None,
    the algorthm divides the data based on the current labels and current features

    """
    if eigen is True:
        if prev_features is not None and prev_labels is not None:
            vector_dict = get_singular_vector(prev_features, prev_labels)
        else:
            vector_dict = get_singular_vector(current_features, current_labels)
    else:
        if prev_features is not None and prev_labels is not None:
            vector_dict = get_mean_vector(prev_features, prev_labels)
        else:
            vector_dict = get_mean_vector(current_features, current_labels)

    scores = get_score(
        vector_dict,
        features=current_features,
        labels=current_labels,
        normalization=norm,
    )

    clean_labels = fit_mixture(scores, current_labels, p_threshold=p_threshold)

    return clean_labels
