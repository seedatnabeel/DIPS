"""
Adapted from: https://github.com/amzn/confident-sinkhorn-allocation

Confident Sinkhorn Allocation for Pseudo-Labeling
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


try:
    from dips_selector import *
    from pseudo_labeling import Pseudo_Labeling
    from fine_method import *

except:
    from .dips_selector import *
    from .pseudo_labeling import Pseudo_Labeling
    from .fine_method import *


# UPS: ===========================================================================================
#  Rizve, Mamshad Nayeem, Kevin Duarte, Yogesh S. Rawat, and Mubarak Shah.
# "In Defense of Pseudo-Labeling: An Uncertainty-Aware Pseudo-label Selection Framework for Semi-Supervised Learning."
# ICLR. 2020.
#  https://arxiv.org/pdf/2101.06329.pdf
class UPS(Pseudo_Labeling):
    # adaptive thresholding

    def __init__(
        self,
        unlabelled_data,
        x_test,
        y_test,
        num_iters=5,
        upper_threshold=0.8,
        lower_threshold=0.2,
        num_XGB_models=10,
        verbose=False,
        IsMultiLabel=False,
        seed=0,
        nest=100,
        xgb_model=True,
    ):
        """
        unlabelled_data      : [N x d] where N is the number of unlabeled data, d is the feature dimension
        x_test               :[N_test x d]
        y_test               :[N_test x 1] for multiclassification or [N_test x K] for multilabel classification
        num_iters            : number of pseudo-iterations, recommended = 5 as in the paper
        upper_threshold      : the upper threshold used for pseudo-labeling, e.g., we assign label if the prob > 0.8
        fraction_allocation  : the faction of label allocation, if fraction_allocation=1, we assign labels to 100% of unlabeled data
        lower_threshold      : lower threshold, used for UPS
        num_XGB_models       : number of XGB models used for UPS and CSA, recommended = 10
        verbose              : verbose
        IsMultiLabel         : False => Multiclassification or True => Multilabel classification
        """

        super().__init__(
            unlabelled_data,
            x_test,
            y_test,
            num_iters=num_iters,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            num_XGB_models=num_XGB_models,
            verbose=verbose,
            IsMultiLabel=IsMultiLabel,
            seed=seed,
            nest=nest,
            xgb_model=xgb_model,
        )

        self.algorithm_name = "UPS"

    def predict(self, X):
        super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def evaluate_performance(self):
        super().evaluate_performance()

    def uncertainty_score(self, matrix_prob):
        return super().uncertainty_score(matrix_prob)

    def get_prob_at_max_class(self, pseudo_labels_prob):
        return super().get_prob_at_max_class(pseudo_labels_prob)

    def get_max_pseudo_point(self, class_freq, current_iter):
        return super().get_max_pseudo_point(class_freq, current_iter)

    def label_assignment_and_post_processing_UPS(
        self,
        pseudo_labels_prob,
        uncertainty_scores,
        X,
        y,
        current_iter=0,
        upper_threshold=None,
    ):
        """
        Given the threshold, we perform label assignment and post-processing

        Args:
            pseudo_labels_prob: predictive prob [N x K] where N is #unlabels, K is #class
            uncertainty_scores    : uncertainty_score of each data point at each class [N x K]
            X: existing pseudo_labeled + labeled data [ N' x d ]
            y: existing pseudo_labeled + labeled data [ N' x 1 ] for multiclassification
            y: existing pseudo_labeled + labeled data [ N' x K ] for multilabel classification

        Output:
            Augmented X = augmented_X + X
            Augmented y = augmented_y + Y
        """

        if self.IsMultiLabel == False:
            # go over each row (data point), only keep the argmax prob
            # because we only allow a single data point to a single class
            max_prob_matrix = self.get_prob_at_max_class(pseudo_labels_prob)
        else:
            # we dont need to get prob at max class for multi-label
            # because a single data point can be assigned to multiple classes
            max_prob_matrix = pseudo_labels_prob

        assigned_pseudo_labels = np.zeros(
            (max_prob_matrix.shape[0], self.nClass)
        ).astype(int)

        MaxPseudoPoint = [0] * self.nClass
        for cc in range(self.nClass):  # loop over each class

            MaxPseudoPoint[cc] = self.get_max_pseudo_point(
                self.label_frequency[cc], current_iter
            )

            idx_sorted = np.argsort(max_prob_matrix[:, cc])[::-1]  # decreasing

            idx_within_prob = np.where(
                max_prob_matrix[idx_sorted, cc] > self.upper_threshold
            )[0]
            idx_within_prob_uncertainty = np.where(
                uncertainty_scores[idx_sorted[idx_within_prob], cc]
                < self.lower_threshold
            )[0]

            # only select upto MaxPseudoPoint[cc] points
            labels_satisfied_threshold = idx_sorted[idx_within_prob_uncertainty][
                : MaxPseudoPoint[cc]
            ]

            assigned_pseudo_labels[labels_satisfied_threshold, cc] = 1

        if self.verbose:
            print("MaxPseudoPoint", MaxPseudoPoint)

        return self.post_processing_and_augmentation(assigned_pseudo_labels, X, y)

    def selector(self, X, y, dips_xthresh, dips_ythresh, method="loss", epochs=100):

        import torch
        import torch.nn as nn
        from datagnosis.plugins.core.datahandler import DataHandler
        from datagnosis.plugins.core.models.simple_mlp import SimpleMLP

        # datagnosis absolute
        from datagnosis.plugins import Plugins

        from .utils import UCI_MLP

        y = y.reshape(-1)
        n_classes = len(np.unique(y))
        datahander = DataHandler(X, y, batch_size=len(y))

        # creating our model object, which we both want to use downstream, but also we will use to judge the hardness of the data points
        # model = SimpleMLP(input_dim = X.shape[1], output_dim=2)

        model = UCI_MLP(num_features=X.shape[1], num_outputs=n_classes)

        # creating our optimizer and loss function objects
        learning_rate = 0.01
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if method == "dips":
            y = y.reshape(-1)
            datahander = DataHandler(X, y, batch_size=32)

            hcm = Plugins().get(
                "dips",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr=learning_rate,
                epochs=epochs,
                num_classes=n_classes,
                logging_interval=1,
            )

        elif method == "loss":
            hcm = Plugins().get(
                "large_loss",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr=learning_rate,
                epochs=epochs,
                num_classes=n_classes,
                logging_interval=1,
            )

        elif method == "filter":
            hcm = Plugins().get(
                "filtering",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr=learning_rate,
                epochs=epochs,
                num_classes=n_classes,
                logging_interval=1,
                total_samples=len(y),
            )

        elif method == "basicfilter":
            hcm = Plugins().get(
                "basicfilter",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                lr=learning_rate,
                epochs=epochs,
                num_classes=n_classes,
                logging_interval=1,
                total_samples=len(y),
            )

        if method != "fine":
            hcm.fit(
                datahandler=datahander,
                use_caches_if_exist=False,
            )

        if method == "dips":
            confidence, dips_xmetric = hcm.scores
            easy_train, ambig_train, hard_train = get_groups(
                confidence=confidence,
                aleatoric_uncertainty=dips_xmetric,
                dips_xthresh=dips_xthresh,
                dips_ythresh=dips_ythresh,
            )
        elif method == "loss":
            scores = hcm.scores
            threshold = np.percentile(scores, 99)
            easy_train = np.where(scores < threshold)[0]
        elif method == "filter" or method == "basicfilter":
            scores = hcm.scores
            easy_train = np.where(scores == 1)[0]

        if method == "fine":
            try:
                easy_train = fine(current_features=X, current_labels=y)
            except:
                easy_train = np.arange(len(y))

        if len(np.unique(y[easy_train])) != len(np.unique(y)):
            # find one id of each unique label and append to easy_train
            for label in np.unique(y):
                easy_train = np.append(easy_train, np.where(y == label)[0][0])

            # remove duplicates in easy_train
            easy_train = np.unique(easy_train)

        if len(easy_train) < len(np.unique(y)):
            easy_train = np.arange(len(y))

        ambig_train, hard_train = [], []
        return easy_train, ambig_train, hard_train

    def fit(
        self,
        X,
        y,
        dips=False,
        dips_free=True,
        dips_metric="aleatoric",
        dips_xthresh=0.15,
        dips_ythresh=0.2,
        method="dips",
        epochs=100,
    ):
        """
        main algorithm to perform pseudo labelling

        Args:
            X: train features [N x d]
            y: train targets [N x 1]

        Output:
            we record the test_accuracy a vector of test accuracy per pseudo-iteration
        """

        print("=====", self.algorithm_name)
        indices_labeled = np.arange(len(y))

        self.nClass = self.get_number_of_labels(y)

        print("len(y)", len(y))

        self.unlabeled_iterdict = {}
        self.selected_unlabelled_idx_iterdict = {}
        self.dips_iterdict = {}
        self.pseudo_iterdict = {}
        self.model_iterdict = {}

        self.label_frequency = self.estimate_label_frequency(y)
        self.indices = []
        self.data = []
        self.list_models = []

        for current_iter in (
            tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)
        ):

            # Fit to data
            if dips:
                assert dips_metric in [
                    "aleatoric",
                    "epistemic",
                    "entropy",
                    "mi",
                ], "Invalid dips metric"
                assert (
                    dips_xthresh >= 0 and dips_xthresh <= 1
                ), "Invalid dips x-threshold"
                assert (
                    dips_ythresh >= 0 and dips_ythresh <= 1
                ), "Invalid dips y-threshold"

                if self.xgb_model:

                    if dips_free:
                        if current_iter == 0:
                            # compute for free with the original fit
                            self.model.fit(X, y)
                    else:
                        # refit the model
                        self.model.fit(X, y)

                    dips_xgb = DIPS_selector(X=X, y=y)

                    for i in range(1, self.nest):
                        # *** Characterize samples ***
                        dips_xgb.on_epoch_end(clf=self.model, iteration=i)

                # alternative selectors
                if method in ["filter", "basicfilter", "loss", "fine"]:
                    easy_train, ambig_train, hard_train = self.selector(
                        X, y, dips_xthresh, dips_ythresh, method, epochs
                    )
                    confidence = 0.8
                else:
                    # dips selector
                    if dips_metric == "aleatoric":
                        dips_xmetric = dips_xgb.aleatoric
                    elif dips_metric == "epistemic":
                        dips_xmetric = dips_xgb.variability
                    elif dips_metric == "entropy":
                        dips_xmetric = dips_xgb.entropy
                    elif dips_metric == "mi":
                        dips_xmetric = dips_xgb.mi

                    confidence = dips_xgb.confidence

                    easy_train, ambig_train, hard_train = get_groups(
                        confidence=confidence,
                        aleatoric_uncertainty=dips_xmetric,
                        dips_xthresh=dips_xthresh,
                        dips_ythresh=dips_ythresh,
                    )

                # among the easy examples, we look at what proportions of D_lab is kept, and what proportions of D_unlab is kept
                proportion_easy_lab = len(
                    np.intersect1d(easy_train, indices_labeled)
                ) / len(indices_labeled)
                # the first is the prop of pseudo-labeled data which are easy (so the thresholding has already been done)
                if (len(X) - len(indices_labeled)) != 0:
                    proportion_easy_unlab_selected = (
                        len(easy_train)
                        - len(np.intersect1d(easy_train, indices_labeled))
                    ) / (len(X) - len(indices_labeled))
                else:
                    proportion_easy_unlab_selected = "NAN"

                self.indices.append([easy_train, ambig_train, hard_train])
                # Fit to data
                if self.xgb_model:
                    self.model.fit(X[easy_train, :], y[easy_train])

                if method not in ["filter", "basicfilter", "loss", "fine"]:
                    self.dips_iterdict[current_iter] = {
                        "easy": easy_train,
                        "ambig": ambig_train,
                        "hard": hard_train,
                        "confidence": confidence,
                        "dips_xmetric": dips_xmetric,
                        "dips": deepcopy(dips_xgb),
                    }

            else:
                # the labeled dataset is used entirely in the supervised loss, and the reset of points in X is unlabeled data which has been pseudo-labeled

                self.indices.append([np.arange(len(y)), [], []])
                if self.xgb_model:
                    self.model.fit(X, y)

            self.data.append({"X_unlab": self.unlabelled_data, "X": X, "y": y})
            self.list_models.append(deepcopy(self.model))

            self.evaluate_performance()

            if self.xgb_model:
                # estimate prob using unlabelled data on M XGB models
                pseudo_labels_prob_list = [0] * self.num_XGB_models
                for mm in range(self.num_XGB_models):
                    self.XGBmodels_list[mm].fit(X, y)  # fit an XGB model
                    pseudo_labels_prob_list[mm] = (
                        self.get_predictive_prob_for_unlabelled_data(
                            self.XGBmodels_list[mm]
                        )
                    )

            else:
                pseudo_labels_prob_list = [0] * self.num_XGB_models
                for mm in range(self.num_XGB_models):
                    pseudo_labels_prob_list[mm] = (
                        self.get_predictive_prob_for_unlabelled_data(self.model)
                    )

            pseudo_labels_prob_list = np.asarray(pseudo_labels_prob_list)
            pseudo_labels_prob = np.mean(pseudo_labels_prob_list, axis=0)

            # calculate uncertainty estimation for each data points at the argmax class
            uncertainty_scores = np.ones((pseudo_labels_prob.shape))
            for ii in range(
                pseudo_labels_prob.shape[0]
            ):  # go over each row (data points)
                idxMax = np.argmax(pseudo_labels_prob[ii, :])
                uncertainty_scores[ii, idxMax] = np.std(
                    pseudo_labels_prob_list[:, ii, idxMax]
                )

            # this steps adds the pseudo-labeled data to the training set
            X, y = self.label_assignment_and_post_processing_UPS(
                pseudo_labels_prob, uncertainty_scores, X, y, current_iter
            )

            try:
                self.model_iterdict[current_iter] = self.model
                self.unlabeled_iterdict[current_iter] = self.unlabelled_data
                self.selected_unlabelled_idx_iterdict[current_iter] = (
                    self.selected_unlabelled_index
                )
                self.pseudo_iterdict[current_iter] = {
                    "X": self.pseudo_X,
                    "y": self.pseudo_y,
                }
            except:
                pass

            if np.sum(self.num_augmented_per_class) == 0:  # no data point is augmented
                return

            if self.verbose:
                print("#added:", self.num_augmented_per_class, " no train data", len(y))

        # evaluate_performance at the last iteration for reporting purpose
        if self.xgb_model:
            self.model.fit(X, y)

        self.evaluate_performance()
