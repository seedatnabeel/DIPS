"""
Adapted from: https://github.com/amzn/confident-sinkhorn-allocation

Confident Sinkhorn Allocation for Pseudo-Labeling
"""

from copy import deepcopy

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from dips_selector import *
    from fine_method import *
except:
    from .dips_selector import *
    from .fine_method import *


class Pseudo_Labeling(object):
    # implementation of the master class for pseudo-labeling
    # this class will be inherited across other subclasses

    def __init__(
        self,
        unlabelled_data,
        x_test,
        y_test,
        num_iters=5,
        upper_threshold=0.8,
        fraction_allocation=1,
        lower_threshold=None,
        num_XGB_models=0,
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

        self.IsMultiLabel = False
        self.algorithm_name = "Pseudo_Labeling"
        self.x_test = x_test
        self.y_test = y_test
        self.x_lab = None
        self.y_lab = None

        self.IsMultiLabel = IsMultiLabel

        # for house keeping and reporting purpose
        self.len_unlabels = []
        self.len_accepted_ttest = []
        self.len_selected = []
        self.num_augmented_per_class = []
        self.nest = nest
        self.seed = seed
        self.xgb_model = xgb_model

        if xgb_model:
            # this is the XGBoost model for multi-class classification
            param = {}
            param["booster"] = "gbtree"
            param["objective"] = "binary:logistic"
            param["verbosity"] = 0
            param["silent"] = 1
            param["n_estimators"] = self.nest
            param["seed"] = self.seed

            # create XGBoost instance with default hyper-parameters
            # xgb = XGBClassifier(**param,use_label_encoder=False)
            xgb = self.get_XGB_model(param)
            self.model = deepcopy(xgb)
        else:
            self.model = get_neural_net()

        self.unlabelled_data = unlabelled_data  # this is a temporary unlabelled data changing in each iteration
        self.verbose = verbose
        self.upper_threshold = upper_threshold
        self.num_iters = num_iters

        if lower_threshold is not None:
            self.lower_threshold = lower_threshold  # this lower threshold is used for UPS algorithm, not the vanilla Pseudo-labeling

        # allow the pseudo-data is repeated, e.g., without removing them after each iteration
        # create a list of all the indices
        self.unlabelled_indices = list(range(unlabelled_data.shape[0]))

        self.selected_unlabelled_index = []

        if self.verbose:
            print(
                "no of unlabelled data:",
                unlabelled_data.shape[0],
                "\t no of test data:",
                x_test.shape[0],
            )

        # Shuffle the indices
        np.random.shuffle(self.unlabelled_indices)
        self.test_acc = []
        self.test_y_values = []
        self.FractionAllocatedLabel = fraction_allocation  # we will allocate labels to 100% of the unlabeled dataset
        self.num_XGB_models = num_XGB_models  # this is the parameter M in our paper

        if num_XGB_models > 1:  # will be used for CSA and UPS
            # for uncertainty estimation
            # generate multiple models
            params = {
                "max_depth": np.arange(3, 20).astype(int),
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "subsample": np.arange(0.5, 1.0, 0.05),
                "colsample_bytree": np.arange(0.4, 1.0, 0.05),
                "colsample_bylevel": np.arange(0.4, 1.0, 0.05),
                "n_estimators": [100, 200, 300, 500, 600, 700, 1000],
            }

            self.XGBmodels_list = [0] * self.num_XGB_models

            param_list = [0] * self.num_XGB_models
            for tt in range(self.num_XGB_models):

                param_list[tt] = {}

                for key in params.keys():

                    mychoice = np.random.choice(params[key])

                    param_list[tt][key] = mychoice
                    param_list[tt]["verbosity"] = 0
                    param_list[tt]["silent"] = 1
                    param_list[tt]["seed"] = tt

                # self.XGBmodels_list[tt] = XGBClassifier(**param_list[tt],use_label_encoder=False)
                self.XGBmodels_list[tt] = self.get_XGB_model(param_list[tt])

    def get_XGB_model(self, param):
        """
        we create the XGB model depending on multiclass or multi-label setting
        Args:
            param: a predefined hyperparameter for XGBmodel

        Output:
            a single XGBClassifier for multiclass
            or
            a single MultiOutputClassifier for multilabel
        """

        if self.IsMultiLabel == False:
            return XGBClassifier(**param, use_label_encoder=False)
        else:
            return MultiOutputClassifier(
                XGBClassifier(**param, use_label_encoder=False)
            )

    def get_predictive_prob_for_unlabelled_data(self, model):
        """
        Compute the predictive probability within [0,1] for unlabelled data given a single XGB model
        Args:
            model: a single XGBmodel

        Output:
            predictive probability matrix [N x K]
        """

        # neural network
        if not self.xgb_model:
            pseudo_labels_prob = model.predict(self.unlabelled_data, verbose=2)

        if self.xgb_model:
            pseudo_labels_prob = model.predict_proba(self.unlabelled_data)

        # number of unlabeled data
        if self.IsMultiLabel == True:
            pseudo_labels_prob = np.asarray(pseudo_labels_prob).T
            pseudo_labels_prob = pseudo_labels_prob[1, :, :]

        return pseudo_labels_prob

    def estimate_label_frequency(self, y):
        """
        estimate the label frequency empirically from the initial labeled data
        Args:
            y: label vector or matrix (multilabel)

        Output:
            Given K the number of labels, it returns a vector of label frequency [1 x K]
        """

        one_hot = False
        if not self.xgb_model:
            if (y.sum(axis=1) - np.ones(y.shape[0])).sum() == 0:
                one_hot = True
                unique, label_frequency = np.unique(
                    np.argmax(y, axis=1), return_counts=True
                )

        if self.IsMultiLabel == False and one_hot == False:
            if len(self.num_augmented_per_class) > 0:
                unique, label_frequency = np.unique(
                    y[np.sum(self.num_augmented_per_class) :], return_counts=True
                )
            else:
                unique, label_frequency = np.unique(y, return_counts=True)
        else:
            label_frequency = np.sum(y, axis=0)
        print(label_frequency)

        if self.verbose:
            print("==label_frequency without adjustment", np.round(label_frequency, 3))

        # smooth the label frequency if the ratio between the max class / min class is significant >5
        # this smoothing is the implementation trick to prevent biased estimation given limited training data
        ratio = np.max(label_frequency) / np.min(label_frequency)
        if ratio > 5:
            label_frequency = (
                label_frequency / np.sum(label_frequency)
                + np.ones(self.nClass) * 1.0 / self.nClass
            )

        return label_frequency / np.sum(label_frequency)

    def evaluate_performance(self):
        """
        evaluate_performance the classification performance
        Store the result into: self.test_acc which is the accuracy for multiclassification \
                                                    or the precision for multilabel classification
        """

        y_test_pred = self.model.predict(self.x_test)

        test_y_values = {
            "y_pred": self.model.predict(self.x_test),
            "y_score": self.model.predict_proba(self.x_test),
            "y_test": self.y_test,
        }

        if not self.xgb_model:
            _, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
            self.test_acc += [test_acc]
            self.test_y_values += [test_y_values]

        if self.IsMultiLabel == False and self.xgb_model:
            test_acc = np.round(
                accuracy_score(y_test_pred, self.y_test) * 100, 2
            )  # round to 2 digits xx.yy %

            if self.verbose:
                print("+++Test Acc: {:.2f}%".format(test_acc))
            self.test_acc += [test_acc]
        elif self.IsMultiLabel == True and self.xgb_model:  # multi-label classification

            # Precision
            prec = (
                sklearn.metrics.precision_score(
                    self.y_test, y_test_pred, average="samples"
                )
                * 100
            )
            prec = np.round(prec, 2)  # round to 2 digits xx.yy %

            self.test_acc += [prec]  # precision score

            if self.verbose:
                print("+++Test Acc: {:.2f}%".format(prec))

    def get_prob_at_max_class(self, pseudo_labels_prob):
        """
        Given the 2d probability matrix [N x K], we get the probability at the maximum index
        Args:
           pseudo_labels_prob: 2d probability matrix [N x K]

        Returns:
           max_prob_matrix: probability at argmax class [N x 1]
        """
        max_prob_matrix = np.zeros((pseudo_labels_prob.shape))
        for ii in range(pseudo_labels_prob.shape[0]):  # loop over each data point
            idxMax = np.argmax(
                pseudo_labels_prob[ii, :]
            )  # find the highest score class
            max_prob_matrix[ii, idxMax] = pseudo_labels_prob[ii, idxMax]
        return max_prob_matrix

    def post_processing_and_augmentation(self, assigned_pseudo_labels, X, y):
        """
        after assigning the pseudo labels in the previous step, we post-process and augment them into X and y
        Args:
            assigned_pseudo_labels: [N x K] matrix where N is the #unlabels and K is the #class
            assigned_pseudo_labels==0 indicates no assignment
            assigned_pseudo_labels==1 indicates assignment.

            X: existing pseudo_labeled + labeled data [ N' x d ]
            y: existing pseudo_labeled + labeled data [ N' x 1 ] for multiclassification
            y: existing pseudo_labeled + labeled data [ N' x K ] for multilabel classification
        Output:
            Augmented X
            Augmented y
        """
        # print("Post-processing and augmentation...")
        # print(assigned_pseudo_labels.shape)
        sum_by_cols = np.sum(assigned_pseudo_labels, axis=1)
        # print(len(sum_by_cols))
        labels_satisfied_threshold = np.where(sum_by_cols > 0)[0]

        self.num_augmented_per_class.append(
            np.sum(assigned_pseudo_labels, axis=0).astype(int)
        )

        if len(labels_satisfied_threshold) == 0:  # no point is selected
            return X, y

        self.selected_unlabelled_index += labels_satisfied_threshold.tolist()

        # print(f"Labels = {len(labels_satisfied_threshold)}")
        # print(self.unlabelled_data.shape, X.shape, y.shape)
        # augment the assigned labels to X and y ==============================================

        if self.xgb_model:
            self.pseudo_X = self.unlabelled_data[labels_satisfied_threshold, :]
        else:
            self.pseudo_X = self.unlabelled_data[labels_satisfied_threshold, :, :, :]

        X = np.vstack((self.pseudo_X, X))
        if self.IsMultiLabel == False:
            self.pseudo_y = np.argmax(
                (assigned_pseudo_labels[labels_satisfied_threshold, :]), axis=1
            ).reshape(-1, 1)

            # check if one-hot encoded data
            one_hot = False
            if not self.xgb_model:
                if (y.sum(axis=1) - np.ones(y.shape[0])).sum() == 0:
                    self.pseudo_y = assigned_pseudo_labels[
                        labels_satisfied_threshold, :
                    ]
                    one_hot = True
                # y = np.argmax(y, axis=1).reshape(-1, 1)

            if one_hot:
                y = np.vstack((self.pseudo_y, y))
            else:
                y = np.vstack((self.pseudo_y, np.array(y).reshape(-1, 1)))

        else:
            self.pseudo_y = assigned_pseudo_labels[labels_satisfied_threshold, :]
            y = np.vstack((self.pseudo_y, np.array(y)))

        if "CSA" in self.algorithm_name:  # book keeping
            self.len_unlabels.append(len(self.unlabelled_data))
            self.len_accepted_ttest.append(assigned_pseudo_labels.shape[0])
            self.len_selected.append(np.sum(self.num_augmented_per_class))

        # remove the selected data from unlabelled data
        self.unlabelled_data = np.delete(
            self.unlabelled_data, np.unique(labels_satisfied_threshold), 0
        )

        # if one_hot:
        #     # print('UPDATE...')
        #     # print(y.shape)
        #     y =
        #     #print(y.shape)

        if not self.xgb_model:
            X, y = shuffle(X, y)

        return X, y

    def label_assignment_and_post_processing(
        self, pseudo_labels_prob, X, y, current_iter=0, upper_threshold=None
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

        if upper_threshold is None:
            upper_threshold = self.upper_threshold

        if (
            "CSA" in self.algorithm_name
        ):  # if using CSA, we dont use the upper threshold
            upper_threshold = 0

        assigned_pseudo_labels = np.zeros(
            (max_prob_matrix.shape[0], self.nClass)
        ).astype(int)

        MaxPseudoPoint = [0] * self.nClass
        for cc in range(self.nClass):  # loop over each class

            MaxPseudoPoint[cc] = self.get_max_pseudo_point(
                self.label_frequency[cc], current_iter
            )

            idx_sorted = np.argsort(max_prob_matrix[:, cc])[::-1]  # decreasing

            temp_idx = np.where(max_prob_matrix[idx_sorted, cc] > upper_threshold)[0]
            labels_satisfied_threshold = idx_sorted[temp_idx]

            # only select upto MaxPseudoPoint[cc] points
            labels_satisfied_threshold = labels_satisfied_threshold[
                : MaxPseudoPoint[cc]
            ]
            assigned_pseudo_labels[labels_satisfied_threshold, cc] = 1

        if self.verbose:
            print("MaxPseudoPoint", MaxPseudoPoint)

        return self.post_processing_and_augmentation(assigned_pseudo_labels, X, y)

    def get_number_of_labels(self, y):
        """
        # given the label y, return the number of classes

        Args:
            y: label vector (for singlelabel) or matrix (for multilabel)

        Output:
            number of classes or number of labels
        """

        # one-hot encoded data
        if not self.xgb_model:
            if (y.sum(axis=1) - np.ones(y.shape[0])).sum() == 0:
                return len(np.unique(np.argmax(y, axis=1)))

        if self.IsMultiLabel == False:
            return len(np.unique(y))
        else:
            return y.shape[1]

    def get_max_pseudo_point(self, fraction_of_class, current_iter):
        """
        We select more points at the begining and less at later stage

        Args:
            fraction_of_class: vector of the frequency of points per class
            current_iter: current iteration  0,1,2...T
        Output:
            number_of_max_pseudo_points: scalar
        """

        LinearRamp = [
            (self.num_iters - ii) / self.num_iters for ii in range(self.num_iters)
        ]
        SumLinearRamp = np.sum(LinearRamp)

        fraction_iter = (self.num_iters - current_iter) / (
            self.num_iters * SumLinearRamp
        )
        MaxPseudoPoint = (
            fraction_iter
            * fraction_of_class
            * self.FractionAllocatedLabel
            * len(self.unlabelled_data)
        )

        return int(np.ceil(MaxPseudoPoint))

    def get_artifacts(self):
        return self.indices, self.data, self.list_models

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

        if method == "diq":
            y = y.reshape(-1)
            datahander = DataHandler(X, y, batch_size=32)

            hcm = Plugins().get(
                "data_iq",
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

        if method == "diq":
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

        self.nClass = self.get_number_of_labels(y)  # 10

        self.label_frequency = self.estimate_label_frequency(y)

        self.unlabeled_iterdict = {}
        self.selected_unlabelled_idx_iterdict = {}
        self.dips_iterdict = {}
        self.pseudo_iterdict = {}
        self.model_iterdict = {}

        # we store the indices of the labeled data to check the proportions later
        # so at the beginning, in begin and full1, this is smaller than len(X_lab).

        indices_labeled = np.arange(len(X))

        print("n iterations", self.num_iters)
        self.indices = []
        self.data = []
        self.list_models = []
        for current_iter in (
            tqdm(range(self.num_iters)) if self.verbose else range(self.num_iters)
        ):
            print("iteration ", current_iter)

            self.selected_unlabelled_index = []
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

                # DON'T REMOVE
                if dips_free:
                    # compute for free with the original fit
                    if current_iter == 0:
                        self.model.fit(X, y)
                else:
                    # refit the model
                    self.model.fit(X, y)

                dips_xgb = DIPS_selector(X=X, y=y)

                for i in range(1, self.nest):
                    # *** Characterize with dips [LINE 2] ***
                    dips_xgb.on_epoch_end(clf=self.model, iteration=i)

                if method in ["filter", "basicfilter", "loss", "fine"]:
                    easy_train, ambig_train, hard_train = self.selector(
                        X, y, dips_xthresh, dips_ythresh, method, epochs
                    )
                    confidence = 0.8
                else:
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

                # if self.xgb_model:
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
                self.indices.append([np.arange(len(y)), [], []])
                if self.xgb_model:
                    self.model.fit(X, y)

            if dips:
                self.data.append(
                    {
                        "X_unlab": self.unlabelled_data,
                        "X": X[easy_train, :],
                        "y": y[easy_train],
                    }
                )
            else:
                self.data.append({"X_unlab": self.unlabelled_data, "X": X, "y": y})

            self.list_models.append(deepcopy(self.model))
            # evaluate_performance the performance on test set after Fit the model given the data
            self.evaluate_performance()

            # Predictive probability on the unlabeled data
            pseudo_labels_prob = self.get_predictive_prob_for_unlabelled_data(
                self.model
            )

            X, y = self.label_assignment_and_post_processing(
                pseudo_labels_prob, X, y, current_iter
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
