"""
Adapted from: https://github.com/amzn/confident-sinkhorn-allocation

Confident Sinkhorn Allocation for Pseudo-Labeling
"""

from copy import deepcopy

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from dips_selector import *
    from pseudo_labeling import Pseudo_Labeling
except:
    from .dips_selector import *
    from .pseudo_labeling import Pseudo_Labeling


# FlexMatch Strategy for Pseudo-Labeling =======================================================================
# Zhang, Bowen, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki.
# "Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling." NeurIPS 2021
class FlexMatch(Pseudo_Labeling):
    # adaptive thresholding

    def __init__(
        self,
        unlabelled_data,
        x_test,
        y_test,
        num_iters=5,
        upper_threshold=0.9,
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
            upper_threshold=upper_threshold,
            verbose=verbose,
            IsMultiLabel=IsMultiLabel,
            seed=seed,
            nest=nest,
        )

        self.algorithm_name = "FlexMatch"

    def predict(self, X):
        super().predict(X)

    def predict_proba(self, X):
        super().predict_proba(X)

    def evaluate_performance(self):
        super().evaluate_performance()

    def get_max_pseudo_point(self, class_freq, current_iter):
        return super().get_max_pseudo_point(class_freq, current_iter)

    def label_assignment_and_post_processing_FlexMatch(
        self, pseudo_labels_prob, X, y, current_iter=0, upper_threshold=None
    ):
        """
        Given the threshold, perform label assignments and augmentation
        This function is particular for FlexMatch
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

        # for each class, count the number of points > threshold
        # this is the technique used in FlexMatch
        countVector = [0] * self.nClass
        for cc in range(self.nClass):
            temp = np.where(max_prob_matrix[:, cc] > self.upper_threshold)[0]
            countVector[cc] = len(temp)
        countVector_normalized = np.asarray(countVector) / np.max(countVector)

        if upper_threshold is None:
            upper_threshold = self.upper_threshold

        # assign labels if the prob > threshold ========================================================
        assigned_pseudo_labels = np.zeros(
            (max_prob_matrix.shape[0], self.nClass)
        ).astype(int)
        MaxPseudoPoint = [0] * self.nClass
        for cc in range(self.nClass):  # loop over each class

            # note that in FlexMatch, the upper_threshold is updated below before using as the threshold
            flex_class_upper_thresh = countVector_normalized[cc] * self.upper_threshold

            # obtain the maximum number of points can be assigned per class
            MaxPseudoPoint[cc] = self.get_max_pseudo_point(
                self.label_frequency[cc], current_iter
            )

            idx_sorted = np.argsort(max_prob_matrix[:, cc])[::-1]  # decreasing

            temp_idx = np.where(
                max_prob_matrix[idx_sorted, cc] > flex_class_upper_thresh
            )[0]
            labels_satisfied_threshold = idx_sorted[temp_idx]

            # only select upto MaxPseudoPoint[cc] points
            labels_satisfied_threshold = labels_satisfied_threshold[
                : MaxPseudoPoint[cc]
            ]
            assigned_pseudo_labels[labels_satisfied_threshold, cc] = 1

        if self.verbose:
            print("MaxPseudoPoint", MaxPseudoPoint)

        # post-processing and augmenting the data into X and Y ==========================================
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
        indices_labeled = np.arange(len(y))

        print("=====", self.algorithm_name)

        self.nClass = self.get_number_of_labels(y)

        self.unlabeled_iterdict = {}
        self.selected_unlabelled_idx_iterdict = {}
        self.dips_iterdict = {}
        self.pseudo_iterdict = {}
        self.model_iterdict = {}

        self.indices = []
        self.data = []
        self.list_models = []
        self.label_frequency = self.estimate_label_frequency(y)

        for current_iter in (
            tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)
        ):

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
                    # compute for free with the original fit
                    if current_iter == 0:
                        self.model.fit(X, y)
                else:
                    # refit the model
                    self.model.fit(X, y)

                dips_xgb = DIPS_selector(X=X, y=y)

                for i in range(1, self.nest):
                    # *** Characterize with dips  ***
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
                self.indices.append([np.arange(len(y)), [], []])
                self.model.fit(X, y)

            self.data.append({"X_unlab": self.unlabelled_data, "X": X, "y": y})
            self.list_models.append(self.model)

            self.evaluate_performance()

            # estimate prob using unlabelled data
            pseudo_labels_prob = self.get_predictive_prob_for_unlabelled_data(
                self.model
            )

            X, y = self.label_assignment_and_post_processing_FlexMatch(
                pseudo_labels_prob, X, y, current_iter=0
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

            if np.sum(self.num_augmented_per_class) == 0:  # no data point is augmented
                return  # self.test_acc

        # evaluate_performance at the last iteration for reporting purpose
        self.model.fit(X, y)

        self.evaluate_performance()
