# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 20:14:22 2022

@author: Vu Nguyen
"""
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def str2num(s, encoder):
    return encoder[s]


def append_acc_early_termination(AccList, NumIter):
    if len(AccList) <= NumIter:
        Acc_Last_Iter = AccList[-1]
        AccList = AccList + [Acc_Last_Iter] * (1 + NumIter - len(AccList))

    return AccList


def rename_dataset(dataset_name):
    print(dataset_name)
    newname = []

    if dataset_name == "madelon_no":
        return "Madelon"
    elif dataset_name == "synthetic_control_6c":
        return "Synthetic Control"
    elif dataset_name == "digits":
        return "Digits"
    elif dataset_name == "analcatdata_authorship":
        return "Analcatdata"
    elif dataset_name == "German-credit":
        return "German Credit"
    elif dataset_name == "segment_2310_20":
        return "Segment"
    elif dataset_name == "wdbc_569_31":
        return "Wdbc"
    elif dataset_name == "dna_no":
        return "Dna"
    elif dataset_name == "agaricus-lepiota":
        return "Agaricus-Lepiota"
    elif dataset_name == "breast_cancer":
        return "Breast Cancer"
    elif dataset_name == "agaricus-lepiota":
        return "Agaricus-Lepiota"
    elif dataset_name == "emotions":
        return "Emotions"


# 18,7,6,4,2
def get_train_test_unlabeled(
    _datasetName, prop_lab, path_to_data, random_state=0
):  # for multi-classification
    """
    path_to_data='all_data.pickle'
    """

    # load the data
    with open(path_to_data, "rb") as handle:
        [all_data, datasetName_list] = pickle.load(handle)

    dataset_index = datasetName_list.index(_datasetName)
    data = all_data[dataset_index]

    # if dataset_index<14:
    if _datasetName in [
        "segment_2310_20",
        "wdbc_569_31",
        "steel-plates-fault",
        "analcatdata_authorship",
        "synthetic_control_6c",
        "vehicle_846_19",
        "German-credit",
        "gina_agnostic_no",
        "madelon_no",
        "texture",
        "gas_drift",
        "dna_no",
    ]:
        _dic = list(set(data.values[:, -1]))
        num_labels = len(_dic)
        encoder = {}
        for i in range(len(_dic)):
            encoder[_dic[i]] = i

        # shuffle original dataset
        data = data.sample(frac=1, random_state=42)
        X = data.values[:, :-1]
        # X = scale(X)  # scale the X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = np.array([str2num(s, encoder) for s in data.values[:, -1]])
    else:
        X = data[:, :-1]
        Y = data[:, -1]

    # if dataset_index in [9,1,16]:
    if _datasetName in ["hill-valley", "gina_agnostic_no", "agaricus-lepiota"]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )

    # elif dataset_index in [17,8]:
    elif _datasetName in ["German-credit", "breast_cancer"]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )

    # elif dataset_index in [18,6,4]:
    elif _datasetName in ["steel-plates-fault", "synthetic_control_6c", "digits"]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )

    # elif dataset_index in [10,15,12,14,11,13]: # label / unlabel > 15:1
    elif _datasetName in [
        "madelon_no",
        "texture",
        "gas_drift",
        "dna_no",
        "car",
        "kr_vs_kp",
    ]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )

    # elif dataset_index in [3,5]: # label / unlabel > 15:1
    elif _datasetName in ["wdbc_569_31", "analcatdata_authorship"]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )

    # elif dataset_index in [7]:
    elif _datasetName in ["vehicle_846_19"]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )

    # elif dataset_index in [2]:
    elif _datasetName in ["segment_2310_20"]:
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, train_size=prop_lab, random_state=random_state
        )
    else:
        print(_datasetName + "is not defined. please check!")

    p = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[p], y_train[p]

    p = np.random.permutation(x_unlabeled.shape[0])
    x_unlabeled, y_unlabeled = x_unlabeled[p], y_unlabeled[p]

    y_test = np.reshape(y_test, (-1, 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_unlabeled = np.reshape(y_unlabeled, (-1, 1))

    return x_train, y_train, x_test, y_test, x_unlabeled, y_unlabeled


def get_train_test_unlabeled_for_multilabel(
    _datasetName, path_to_data="all_data_multilabel.pickle", random_state=0
):  # for multi-label classification
    """
    path_to_data='all_data.pickle'
    """

    # load the data
    with open(path_to_data, "rb") as handle:
        [all_data, datasetName_list] = pickle.load(handle)

    dataset_index = datasetName_list.index(_datasetName)
    data = all_data[dataset_index]

    X = data["data"]
    Y = data["target"]

    if _datasetName == "emotions":  # emotions dataset
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.1, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, test_size=0.5, random_state=random_state
        )
    elif _datasetName == "genbase":  # genbase dataset
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.1, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, test_size=0.7, random_state=random_state
        )
    elif _datasetName == "yeast":  # yeast dataset
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=random_state
        )

        x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
            x_train, y_train, test_size=0.7, random_state=random_state
        )
    else:
        print(_datasetName + "is not defined. please check!")

    p = np.random.permutation(x_train.shape[0])
    x_train, y_train = x_train[p], y_train[p]

    p = np.random.permutation(x_unlabeled.shape[0])
    x_unlabeled, y_unlabeled = x_unlabeled[p], y_unlabeled[p]

    return x_train, y_train, x_test, y_test, x_unlabeled


def process_results(results_list, numIters, end_score=True, extracted_iteration=0):
    """Process the results returned by run_baseline

    Args:
        results_list (list): list, of length the number of trials
        numIters (int): number of iterations for the semi-supervised algorithms
        end_score (bool, optional): whether or not to use the accuracy at the last iteration. Defaults to True.

    Returns:
        res: a dictionary with the mean results and se.
    """
    import pandas as pd
    from scipy.stats import sem

    df = pd.DataFrame(results_list)
    res = {}

    for model in list(df.columns):
        metrics = {}

        if model == "supervised_learning_accuracy":
            metric = df["supervised_learning_accuracy"].values
            metrics[f"acc_mean"] = np.nanmean(metric)
            metrics[f"acc_se"] = sem(metric)

        elif model == "fully_supervised_learning_accuracy":
            metric = df["fully_supervised_learning_accuracy"].values
            metrics[f"acc_mean"] = np.nanmean(metric)
            metrics[f"acc_se"] = sem(metric)

        elif model == "supervised_learning_accuracy_easy":
            metric = df["supervised_learning_accuracy_easy"].values
            metrics[f"acc_mean"] = np.nanmean(metric)
            metrics[f"acc_se"] = sem(metric)

        elif model == "full_supervised_learning_accuracy_easy":
            metric = df["full_supervised_learning_accuracy_easy"].values
            metrics[f"acc_mean"] = np.nanmean(metric)
            metrics[f"acc_se"] = sem(metric)

        elif model in ["ups", "pseudo", "csa", "sla", "flex"]:
            for key in (
                df[model].values[0].keys()
            ):  # This loops through vanilla, diq etc.
                if (
                    key == "vanilla"
                    or key == "diq_full"
                    or key == "diq_begin"
                    or key == "diq_full2"
                ):
                    metric = [mydict[key] for mydict in df[model].values]
                    if model == "csa":
                        for idx, iter_list in enumerate(metric):
                            iter_list = list(iter_list)
                            if len(iter_list) != numIters + 1:
                                extra_needed = numIters + 1 - len(iter_list)
                                extra = [iter_list[-1]] * extra_needed
                                iter_list.extend(extra)
                                metric[idx] = np.array(iter_list)

                    metric = [
                        inner_list for inner_list in metric if len(inner_list) > 1
                    ]

                    metric = np.array(metric)

                    if end_score:
                        metrics[f"{key}_mean"] = np.nanmean(metric[:, -1])
                        metrics[f"{key}_se"] = sem(metric[:, -1])
                    else:
                        metrics[f"{key}_mean"] = np.nanmean(
                            metric[:, extracted_iteration]
                        )
                        metrics[f"{key}_se"] = sem(metric[:, extracted_iteration])

        res[model] = metrics

    return res
