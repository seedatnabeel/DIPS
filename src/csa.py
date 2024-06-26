"""
Adapted from: https://github.com/amzn/confident-sinkhorn-allocation

Confident Sinkhorn Allocation for Pseudo-Labeling
"""

import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from dips_selector import *
    from pseudo_labeling import Pseudo_Labeling
except:
    from .dips_selector import *
    from .pseudo_labeling import Pseudo_Labeling


# Confident Sinkhorn Allocation==================================================================================================
class CSA(Pseudo_Labeling):
    def __init__(
        self,
        unlabelled_data,
        x_test,
        y_test,
        num_iters=5,
        num_XGB_models=20,
        confidence_choice="ttest",
        verbose=False,
        IsMultiLabel=False,
        seed=0,
        nest=100,
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
            num_XGB_models=num_XGB_models,
            verbose=verbose,
            IsMultiLabel=IsMultiLabel,
            seed=seed,
            nest=nest,
        )

        self.confidence_choice = confidence_choice

        if self.IsMultiLabel == True:
            # by default, we use total_variance as the main criteria for multilabel classification
            if self.confidence_choice is not None:
                self.confidence_choice = "variance"

        if self.confidence_choice is None or self.confidence_choice == "None":
            self.algorithm_name = "SLA"
        else:
            self.algorithm_name = "CSA_" + self.confidence_choice

        self.elapse_xgb = []
        self.elapse_ttest = []
        self.elapse_sinkhorn = []

        if self.verbose:
            print("number of used XGB models  M=", self.num_XGB_models)

    def predict(self, X):
        super().predict(X)

    def predict_proba(self, X):
        super().predict_proba(X)

    def evaluate_performance(self):
        super().evaluate_performance()

    def get_max_pseudo_point(self, class_freq, current_iter):
        return super().get_max_pseudo_point(class_freq, current_iter)

    def set_ot_regularizer(self, nRow, nCol):
        """
        We set the Sinkhorn regularization parameter based on the ratio of Row/Column

        Args:
            nRow: number of rows in our cost matrix for Sinkhorn algorithm
            nCol: number of columns

        Output:
            regularization
        """

        if nRow / nCol >= 300:
            regulariser = 1
        if nRow / nCol >= 200:
            regulariser = 0.5
        elif nRow / nCol >= 100:
            regulariser = 0.2
        elif nRow / nCol >= 50:
            regulariser = 0.1
        else:
            regulariser = 0.05

        if self.IsMultiLabel:
            if self.nClass > 20:
                regulariser = regulariser * 5
            else:
                regulariser = regulariser * 200

        return regulariser

    def data_uncertainty(self, pseudo_labels_prob_list):
        """
        Args:
            pseudo_labels_prob_list: [M x N x K]
        Output:
            entropy: [N x 1]
        """

        ent = np.zeros(
            (pseudo_labels_prob_list.shape[0], pseudo_labels_prob_list.shape[1])
        )
        for mm in range(pseudo_labels_prob_list.shape[0]):
            ent[mm, :] = self.entropy_prediction(pseudo_labels_prob_list[mm, :, :])

        return np.mean(ent, axis=0)

    def entropy_prediction(self, ave_pred, atClass=None):
        """
        Args:
            ave_pred: [N x K]
        Output:
            entropy: [N x 1]
        """

        ent = [0] * ave_pred.shape[0]

        for ii in range(ave_pred.shape[0]):
            ent[ii] = -np.sum(ave_pred[ii, :] * np.log(ave_pred[ii, :]))
        return np.asarray(ent)

    def total_entropy(self, pseudo_labels_prob_list, atClass=None):
        """
        calculate total entropy
        Args:
            pseudo_labels_prob_list: [M x N x K]: M #XGB, N #unlabels, K #class
        Output:
            total_entropy score [N x 1]
        """

        ave_pred = np.mean(pseudo_labels_prob_list, axis=0)  # average over model

        total_uncertainty = self.entropy_prediction(ave_pred, atClass)
        return total_uncertainty

    def knowledge_uncertainty(self, pred):

        total_uncertainty = self.total_uncertainty(pred)

        data_uncertainty = self.data_uncertainty(pred)

        knowledge_uncertainty = total_uncertainty - data_uncertainty
        return knowledge_uncertainty

    def total_variance(self, pseudo_labels_prob_list):
        """
        calculate total variance
        Args:
            pseudo_labels_prob_list: [M x N x K]: M #XGB, N #unlabels, K #class
        Output:
            standard deviation score [N x 1]
        """

        # [nModel, nPoint, nClass]
        std_pred = np.std(pseudo_labels_prob_list, axis=0)  # std over models
        total_std = np.sum(std_pred, axis=1)  # sum of std over classes

        return total_std

    def calculate_ttest(self, pseudo_labels_prob_list):
        """
        calculate t-test
        Args:
            pseudo_labels_prob_list: [M x N x K]: M #XGB, N #unlabels, K #class
        Output:
            t-test score [N x 1]
        """

        num_points = pseudo_labels_prob_list.shape[1]

        var_rows_argmax = [0] * num_points
        var_rows_arg2ndmax = [0] * num_points

        t_test = [0] * num_points
        t_value = [0] * num_points

        pseudo_labels_prob = np.mean(pseudo_labels_prob_list, axis=0)

        temp = np.argsort(-pseudo_labels_prob, axis=1)  # decreasing
        idxargmax = temp[:, 0]
        idx2nd_argmax = temp[:, 1]

        for jj in range(num_points):  # go over each row (data points)

            idxmax = idxargmax[jj]
            idx2ndmax = idx2nd_argmax[jj]

            var_rows_argmax[jj] = np.var(pseudo_labels_prob_list[:, jj, idxmax])
            var_rows_arg2ndmax[jj] = np.var(pseudo_labels_prob_list[:, jj, idx2ndmax])

            nominator = (
                pseudo_labels_prob[jj, idxmax] - pseudo_labels_prob[jj, idx2ndmax]
            )
            temp = (
                0.1 + var_rows_argmax[jj] + var_rows_arg2ndmax[jj]
            ) / self.num_XGB_models
            denominator = np.sqrt(temp)
            t_test[jj] = nominator / denominator

            # compute degree of freedom=========================================
            nominator = (var_rows_argmax[jj] + var_rows_arg2ndmax[jj]) ** 2

            denominator = var_rows_argmax[jj] ** 2 + var_rows_arg2ndmax[jj] ** 2
            denominator = denominator / (self.num_XGB_models - 1)
            dof = nominator / denominator

            t_value[jj] = stats.t.ppf(1 - 0.025, dof)

            t_test[jj] = t_test[jj] - t_value[jj]

        return t_test

    def label_assignment_and_post_processing_for_CSA(
        self, assignment_matrix, pseudo_labels_prob, X, y, current_iter=0
    ):
        """
        Given the threshold, we perform label assignment and post-processing

        Args:
            pseudo_labels_prob: predictive prob [N x K] where N is #unlabels, K is #class
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

        assignment_matrix = self.get_prob_at_max_class(assignment_matrix)

        assigned_pseudo_labels = np.zeros(
            (max_prob_matrix.shape[0], self.nClass)
        ).astype(int)

        MaxPseudoPoint = [0] * self.nClass
        for cc in range(self.nClass):  # loop over each class

            MaxPseudoPoint[cc] = self.get_max_pseudo_point(
                self.label_frequency[cc], current_iter
            )

            idx_sorted = np.argsort(assignment_matrix[:, cc])[::-1]  # decreasing

            idx_assignment = np.where(assignment_matrix[idx_sorted, cc] > 0)[0]

            # we dont accept labels with less than 0.5 prediction, this works well for multilabel classification
            idx_satisfied = np.where(
                pseudo_labels_prob[idx_sorted[idx_assignment], cc] > 0.5
            )[0]

            # only select upto MaxPseudoPoint[cc] points
            labels_satisfied_threshold = idx_sorted[idx_satisfied][: MaxPseudoPoint[cc]]

            assigned_pseudo_labels[labels_satisfied_threshold, cc] = 1

        if self.verbose:
            print("MaxPseudoPoint", MaxPseudoPoint)

        return self.post_processing_and_augmentation(assigned_pseudo_labels, X, y)

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

        self.label_frequency = self.estimate_label_frequency(y)

        self.unlabeled_iterdict = {}
        self.selected_unlabelled_idx_iterdict = {}
        self.dips_iterdict = {}
        self.pseudo_iterdict = {}
        self.model_iterdict = {}

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

                if dips_free:
                    if current_iter == 0:
                        self.model.fit(X, y)
                else:
                    self.model.fit(X, y)

                dips_xgb = DIPS_selector(X=X, y=y)

                for i in range(1, self.nest):
                    # *** Characterize with dips [LINE 2] ***
                    dips_xgb.on_epoch_end(clf=self.model, iteration=i)

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

                indices_easy_labeled = np.intersect1d(easy_train, indices_labeled)
                indices_easy_non_labeled = np.setdiff1d(easy_train, indices_labeled)

                self.indices.append([easy_train, ambig_train, hard_train])

                # Fit to data
                self.model.fit(X[easy_train, :], y[easy_train])

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
                self.model.fit(X, y)

            self.data.append({"X_unlab": self.unlabelled_data, "X": X, "y": y})
            self.list_models.append(self.model)

            self.evaluate_performance()

            num_points = self.unlabelled_data.shape[0]
            pseudo_labels_prob_list = [0] * self.num_XGB_models

            tic = time.perf_counter()

            # estimate prob using unlabelled data on M XGB models
            pseudo_labels_prob_list = [0] * self.num_XGB_models
            for mm in range(self.num_XGB_models):
                self.XGBmodels_list[mm].fit(X, y)
                pseudo_labels_prob_list[mm] = (
                    self.get_predictive_prob_for_unlabelled_data(
                        self.XGBmodels_list[mm]
                    )
                )

            toc = time.perf_counter()
            self.elapse_xgb.append(toc - tic)

            pseudo_labels_prob_list = np.asarray(
                pseudo_labels_prob_list
            )  # P [M x N x K]
            pseudo_labels_prob = np.mean(
                pseudo_labels_prob_list, axis=0
            )  # \bar{P} [N x K]

            tic = time.perf_counter()  # Start Time

            # estimate confidence level here====================================
            if self.confidence_choice == "variance":
                tot_variance = self.total_variance(pseudo_labels_prob_list)
                confidence = 1 - tot_variance
                confidence = confidence - np.mean(confidence)
            elif self.confidence_choice == "neg_variance":
                confidence = self.total_variance(pseudo_labels_prob_list)
                confidence = confidence - np.mean(confidence)
            elif self.confidence_choice == "entropy":
                tot_ent = self.total_entropy(pseudo_labels_prob_list)
                confidence = 1 - tot_ent
                confidence = confidence - 0.5 * np.mean(confidence)
            elif self.confidence_choice == "neg_entropy":
                confidence = self.total_entropy(pseudo_labels_prob_list)
                confidence = confidence - np.mean(confidence)

            elif self.confidence_choice == "ttest":
                confidence = self.calculate_ttest(pseudo_labels_prob_list)
            elif self.confidence_choice == "neg_ttest":
                confidence = self.calculate_ttest(pseudo_labels_prob_list)
                confidence = -np.asarray(confidence)
            elif (
                self.confidence_choice == None or self.confidence_choice == "None"
            ):  # not using any confidence score, accepting all data point similar to SLA
                confidence = np.ones((1, num_points))

            confidence = np.clip(confidence, a_min=0, a_max=np.max(confidence))

            toc = time.perf_counter()  # End Time
            self.elapse_ttest.append(toc - tic)

            # for numerical stability of OT, select the nonzero entry only
            idxNoneZero = np.where(confidence > 0)[0]
            # idxNoneZero=np.where( (confidence>0) & (confidence<0.9*np.max(confidence)) )[0]
            num_points = len(idxNoneZero)

            if self.verbose:
                print(
                    "num_points accepted= ",
                    num_points,
                    " total num_points=",
                    len(self.unlabelled_data),
                )

            if (
                len(idxNoneZero) == 0
            ):  # terminate if could not find any point satisfying constraints
                return self.test_acc

            # Sinkhorn's algorithm ======================================================================
            # fraction of label being assigned.
            max_allocation_point = self.get_max_pseudo_point(
                class_freq=1, current_iter=current_iter
            )
            rho = max_allocation_point / len(self.unlabelled_data)

            # regulariser for Sinkhorn's algorithm
            regulariser = self.set_ot_regularizer(num_points, self.nClass)

            tic = time.perf_counter()

            # this is w_{+} and w_{-} in the paper
            upper_b_per_class = self.label_frequency * 1.1
            lower_b_per_class = self.label_frequency * 0.9

            # we define row marginal distribution =============================
            row_marginal = np.ones(num_points)
            temp = (
                num_points
                * rho
                * (np.sum(upper_b_per_class) - np.sum(lower_b_per_class))
            )
            row_marginal = np.append(row_marginal, temp)

            if self.verbose:
                print(
                    "#unlabel={:d} #points/#classes={:d}/{:d}={:.2f} reg={:.2f}".format(
                        len(self.unlabelled_data),
                        num_points,
                        self.nClass,
                        num_points / self.nClass,
                        regulariser,
                    )
                )

            C = 1 - pseudo_labels_prob  # cost # expand Cost matrix
            C = C[idxNoneZero, :]

            C = np.vstack((C, np.zeros((1, self.nClass))))
            C = np.hstack((C, np.zeros((len(idxNoneZero) + 1, 1))))

            K = np.exp(-C / regulariser)

            # define column marginal distribution ==============================
            col_marginal = (
                rho * upper_b_per_class * num_points
            )  # frequency of the class label
            temp = num_points * (1 - rho * np.sum(lower_b_per_class))
            col_marginal = np.append(col_marginal, temp)

            # checking the total mass of column marginal ~ row marginal
            if np.abs(np.sum(col_marginal) - np.sum(row_marginal)) > 0.001:
                print("np.sum(dist_labels) - np.sum(dist_points) > 0.001")

            # initialize uu and perform Sinkhorn algorithm
            uu = np.ones((num_points + 1,))
            for jj in range(100):
                vv = col_marginal / np.dot(K.T, uu)
                uu = row_marginal / np.dot(K, vv)

            # compute label assignment matrix Q'
            Q_prime = np.atleast_2d(uu).T * (K * vv.T)

            toc = time.perf_counter()
            self.elapse_sinkhorn.append(toc - tic)

            # this is the final Q matrix
            assignment_matrix_Q = np.zeros((pseudo_labels_prob.shape))
            assignment_matrix_Q[idxNoneZero, :] = Q_prime[:-1, :-1]

            X, y = self.label_assignment_and_post_processing_for_CSA(
                assignment_matrix_Q, pseudo_labels_prob, X, y, current_iter
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

            if self.verbose:
                print(
                    "#augmented:",
                    self.num_augmented_per_class,
                    " len of training data ",
                    len(y),
                )

        # evaluate_performance at the last iteration for reporting purpose
        self.model.fit(X, y)

        self.evaluate_performance()
