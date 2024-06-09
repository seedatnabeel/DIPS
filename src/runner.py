import sys

sys.path.append("..")

from copy import deepcopy
from random import sample

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from baseline_functions import *
    from dips_selector import *
    from data_loader import *
    from utils import (
        append_acc_early_termination,
        get_train_test_unlabeled,
        get_train_test_unlabeled_for_multilabel,
    )
except:
    from .baseline_functions import *
    from .dips_selector import *
    from .data_loader import *
    from .utils import (
        append_acc_early_termination,
        get_train_test_unlabeled,
        get_train_test_unlabeled_for_multilabel,
    )


def run_baselines(
    numTrials,
    numIters,
    upper_threshold,
    num_XGB_models,
    nest,
    seed,
    dataset_name,
    dips_metric="aleatoric",
    dips_xthresh=0.15,
    dips_ythresh=0.2,
    prop_data=1,
    prop_lab=0.1,
    subsample_size=1,
    loss=False,
    verbose=False,
    algorithm_list=[
        "Supervised_Learning",
        "Pseudo_Labeling",
        "FlexMatch",
        "UPS",
        "SLA",
        "CSA",
    ],
):

    overall_result_dicts = []
    overall_data_dicts = []
    overall_model_dicts = []

    for i in tqdm(range(numTrials)):
        try:
            seed = seed + i
            seed = seed * 100

            print(f"Trial {i+1}/{numTrials}")
            results = {}
            data = {}
            models = {}

            print("Loading data...")
            if dataset_name in [
                "seer",
                "cutract",
                "covid",
                "support",
                "adult",
                "bank",
                "drug",
                "metabric",
                "fraud",
                "maggic",
            ]:
                df_feat, df_label, df = get_data(dataset=dataset_name, prop=prop_data)

                x_train, x_test, y_train, y_test = train_test_split(
                    df_feat, df_label, test_size=0.2, random_state=seed
                )

                x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(
                    x_train, y_train, train_size=prop_lab, random_state=seed
                )

                if subsample_size < 1:
                    x_train, _, y_train, _ = train_test_split(
                        x_train, y_train, train_size=subsample_size, random_state=seed
                    )

            elif dataset_name in ["two_moons"]:
                n_total = 1000
                n_labeled = int(prop_lab * n_total)
                n_unlabeled = n_total - n_labeled
                n_test = 10000
                x_unlabeled, y_unlabeled, x_train, y_train, x_test, y_test = two_moons(
                    n_unlabeled, n_labeled, n_test, noise=0.4, random_state=42
                )

            else:
                path_to_file = "./data/all_data.pickle"
                (
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    x_unlabeled,
                    y_unlabeled,
                ) = get_train_test_unlabeled(
                    dataset_name, prop_lab, path_to_file, random_state=seed
                )

            datasize = x_train.shape

            total_samples = len(x_train) + len(x_test) + len(x_unlabeled)

            print(
                f"# total samples = {total_samples} ({prop_lab} - prop of lab among training data)"
            )

            print(f"# training points = {y_train.shape[0]}")

            print(f"# test points = {y_test.shape[0]}")

            print(f"# unlabelled points = {x_unlabeled.shape[0]}")

            x_unlabeled, x_test, y_test, x_train, y_train = (
                np.asarray(x_unlabeled),
                np.asarray(x_test),
                np.asarray(y_test),
                np.asarray(x_train),
                np.asarray(y_train),
            )

            # # Supervised learning - Train an XGBoost model
            param = {}
            param["booster"] = "gbtree"
            param["objective"] = "binary:logistic"
            param["verbosity"] = 0
            param["n_estimators"] = nest
            param["silent"] = 1
            param["seed"] = seed

            ############################################################
            ### STORE THE DATA FOR POST-HOC TSNE PLOTS ##################
            ############################################################
            data["plain_data"] = {
                "x_lab": x_train,
                "y_lab": y_train,
                "x_unlab": x_unlabeled,
                "y_unlab": y_unlabeled,
                "x_test": x_test,
                "y_test": y_test,
            }

            ########################################################
            #### UPPER BOUND ########################################
            #### THIS PART USES ALL THE LABELS OF THE UNLABELED DATA
            ############################################################

            print("Training Fully Supervised model...")
            # create XGBoost instance with default hyper-parameters
            xgb = XGBClassifier(**param)
            all_x = np.concatenate((x_train, x_unlabeled))
            all_y = np.concatenate((y_train, y_unlabeled))

            xgb.fit(all_x, all_y)

            # evaluate the performance on the test set
            y_test_pred = xgb.predict(x_test)
            fully_supervised_learning_accuracy = np.round(
                accuracy_score(y_test_pred, y_test) * 100, 2
            )  # round to 2 digits xx.yy %

            results["fully_supervised_learning_accuracy"] = (
                fully_supervised_learning_accuracy
            )

            # Now preprocess all the data, to get the easy examples

            # Run dips
            # Note that we use all_x and all_y here
            dips_xgb = DIPS_selector(X=all_x, y=all_y)

            for i in range(1, nest):
                # *** Characterize with dips [LINE 2] ***
                dips_xgb.on_epoch_end(clf=xgb, iteration=i)

            # *** Access metrics ***
            if dips_metric == "aleatoric":
                dips_xmetric = dips_xgb.aleatoric
            elif dips_metric == "epistemic":
                dips_xmetric = dips_xgb.variability
            elif dips_metric == "entropy":
                dips_xmetric = dips_xgb.entropy
            elif dips_metric == "mi":
                dips_xmetric = dips_xgb.mi

            confidence = dips_xgb.confidence

            # adaptive threshold
            dips_xthresh = 0.75 * (np.max(dips_xmetric) - np.min(dips_xmetric))

            easy_train, ambig_train, hard_train = get_groups(
                confidence=confidence,
                aleatoric_uncertainty=dips_xmetric,
                dips_xthresh=dips_xthresh,
                dips_ythresh=dips_ythresh,
            )

            # Now train with the easy examples
            print("Training Preprocess + Full Supervised model...")
            # create XGBoost instance with default hyper-parameters
            xgb = XGBClassifier(**param)

            # if x_train is not a numpy array, it is a dataframe
            if type(all_x) is not np.ndarray:
                # fit with the easy training data
                xgb.fit(all_x.iloc[easy_train], all_y.iloc[easy_train])
            else:
                # fit with the easy training data
                xgb.fit(all_x[easy_train], all_y[easy_train])

            # evaluate the performance on the test set
            y_test_pred = xgb.predict(x_test)
            full_supervised_learning_accuracy_easy = np.round(
                accuracy_score(y_test_pred, y_test) * 100, 2
            )  # round to 2 digits xx.yy %

            results["full_supervised_learning_accuracy_easy"] = (
                full_supervised_learning_accuracy_easy
            )

            ########################################################
            ### NOW ONTO THE NORMAL SUPERVISED SETTING ##############
            ########################################################

            print("Training Supervised model...")
            # create XGBoost instance with default hyper-parameters
            xgb = XGBClassifier(**param)

            xgb.fit(x_train, y_train)

            # evaluate the performance on the test set
            y_test_pred = xgb.predict(x_test)
            supervised_learning_accuracy = np.round(
                accuracy_score(y_test_pred, y_test) * 100, 2
            )  # round to 2 digits xx.yy %

            results["supervised_learning_accuracy"] = supervised_learning_accuracy

            # Run dips
            dips_xgb = DIPS_selector(X=x_train, y=y_train)

            for i in range(1, nest):
                # *** Characterize with dips [LINE 2] ***
                dips_xgb.on_epoch_end(clf=xgb, iteration=i)

            # *** Access metrics ***
            if dips_metric == "aleatoric":
                dips_xmetric = dips_xgb.aleatoric
            elif dips_metric == "epistemic":
                dips_xmetric = dips_xgb.variability
            elif dips_metric == "entropy":
                dips_xmetric = dips_xgb.entropy
            elif dips_metric == "mi":
                dips_xmetric = dips_xgb.mi

            confidence = dips_xgb.confidence

            assert len(confidence) == len(y_train)

            # adaptive threshold
            dips_xthresh = 0.75 * (np.max(dips_xmetric) - np.min(dips_xmetric))

            easy_train, ambig_train, hard_train = get_groups(
                confidence=confidence,
                aleatoric_uncertainty=dips_xmetric,
                dips_xthresh=dips_xthresh,
                dips_ythresh=dips_ythresh,
            )

            print("Training Preprocess + Supervised model...")
            # create XGBoost instance with default hyper-parameters
            xgb = XGBClassifier(**param)

            if type(x_train) is not np.ndarray:
                # fit with the easy training data
                xgb.fit(x_train.iloc[easy_train], y_train.iloc[easy_train])
            else:
                # fit with the easy training data
                xgb.fit(x_train[easy_train], y_train[easy_train])

            # evaluate the performance on the test set
            y_test_pred = xgb.predict(x_test)
            supervised_learning_accuracy_easy = np.round(
                accuracy_score(y_test_pred, y_test) * 100, 2
            )  # round to 2 digits xx.yy %

            results["supervised_learning_accuracy_easy"] = (
                supervised_learning_accuracy_easy
            )

            if "UPS" in algorithm_list:
                print("Running UPS...")
                (
                    ups_acc_vanilla,
                    ups_acc_dips_begin,
                    ups_acc_dips_full,
                    ups_acc_dips_partial,
                    artifacts,
                ) = run_UPS(
                    x_unlabeled=x_unlabeled,
                    x_test=x_test,
                    y_test=y_test,
                    x_train=x_train,
                    y_train=y_train,
                    numIters=numIters,
                    num_XGB_models=num_XGB_models,
                    nest=nest,
                    seed=seed,
                    easy_train=easy_train,
                    dips_metric=dips_metric,
                    dips_xthresh=dips_xthresh,
                    dips_ythresh=dips_ythresh,
                    verbose=verbose,
                )

                results["ups"] = {
                    "vanilla": ups_acc_vanilla,
                    "dips_begin": ups_acc_dips_begin,
                    "dips_full": ups_acc_dips_full,
                    "dips_partial": ups_acc_dips_partial,
                }

                data["ups"] = {
                    "vanilla": artifacts["vanilla"]["data"],
                    "dips_begin": artifacts["begin"]["data"],
                    "dips_full": artifacts["full1"]["data"],
                    "dips_partial": artifacts["partial"]["data"],
                }
                models["ups"] = {
                    "vanilla": artifacts["vanilla"]["models"],
                    "dips_begin": artifacts["begin"]["models"],
                    "dips_full": artifacts["full1"]["models"],
                    "dips_partial": artifacts["partial"]["models"],
                }

            if "Pseudo_Labeling" in algorithm_list:

                print("Running Pseudo Labeling...")

                (
                    pseudo_labeling_acc_vanilla,
                    pseudo_labeling_acc_dips_begin,
                    pseudo_labeling_acc_dips_full,
                    pseudo_labeling_acc_dips_partial,
                    artifacts,
                ) = run_pseudo(
                    x_unlabeled=x_unlabeled,
                    x_test=x_test,
                    y_test=y_test,
                    x_train=x_train,
                    y_train=y_train,
                    numIters=numIters,
                    upper_threshold=upper_threshold,
                    nest=nest,
                    seed=seed,
                    easy_train=easy_train,
                    dips_metric=dips_metric,
                    dips_xthresh=dips_xthresh,
                    dips_ythresh=dips_ythresh,
                    verbose=verbose,
                )

                results["pseudo"] = {
                    "vanilla": pseudo_labeling_acc_vanilla,
                    "dips_begin": pseudo_labeling_acc_dips_begin,
                    "dips_full": pseudo_labeling_acc_dips_full,
                    "dips_partial": pseudo_labeling_acc_dips_partial,
                }

                data["pseudo"] = {
                    "vanilla": artifacts["vanilla"]["data"],
                    "dips_begin": artifacts["begin"]["data"],
                    "dips_full": artifacts["full1"]["data"],
                    "dips_partial": artifacts["partial"]["data"],
                }

                models["pseudo"] = {
                    "vanilla": artifacts["vanilla"]["models"],
                    "dips_begin": artifacts["begin"]["models"],
                    "dips_full": artifacts["full1"]["models"],
                    "dips_partial": artifacts["partial"]["models"],
                }

            if "CSA" in algorithm_list:
                print("Running CSA...")
                (
                    csa_acc_vanilla,
                    csa_acc_dips_begin,
                    csa_acc_dips_full,
                    csa_acc_dips_partial,
                    artifacts,
                ) = run_CSA(
                    x_unlabeled=x_unlabeled,
                    x_test=x_test,
                    y_test=y_test,
                    x_train=x_train,
                    y_train=y_train,
                    numIters=numIters,
                    num_XGB_models=num_XGB_models,
                    nest=nest,
                    seed=seed,
                    easy_train=easy_train,
                    dips_metric=dips_metric,
                    dips_xthresh=dips_xthresh,
                    dips_ythresh=dips_ythresh,
                    verbose=verbose,
                )

                results["csa"] = {
                    "vanilla": csa_acc_vanilla,
                    "dips_begin": csa_acc_dips_begin,
                    "dips_full": csa_acc_dips_full,
                    "dips_partial": csa_acc_dips_partial,
                }

                data["csa"] = {
                    "vanilla": artifacts["vanilla"]["data"],
                    "dips_begin": artifacts["begin"]["data"],
                    "dips_full": artifacts["full1"]["data"],
                    "dips_partial": artifacts["partial"]["data"],
                }
                models["csa"] = {
                    "vanilla": artifacts["vanilla"]["models"],
                    "dips_begin": artifacts["begin"]["models"],
                    "dips_full": artifacts["full1"]["models"],
                    "dips_partial": artifacts["partial"]["models"],
                }

            if "SLA" in algorithm_list:
                print("Running SLA...")
                (
                    sla_acc_vanilla,
                    sla_acc_dips_begin,
                    sla_acc_dips_full,
                    sla_acc_dips_partial,
                    artifacts,
                ) = run_SLA(
                    x_unlabeled=x_unlabeled,
                    x_test=x_test,
                    y_test=y_test,
                    x_train=x_train,
                    y_train=y_train,
                    numIters=numIters,
                    num_XGB_models=num_XGB_models,
                    nest=nest,
                    seed=seed,
                    easy_train=easy_train,
                    dips_metric=dips_metric,
                    dips_xthresh=dips_xthresh,
                    dips_ythresh=dips_ythresh,
                    verbose=verbose,
                )

                results["sla"] = {
                    "vanilla": sla_acc_vanilla,
                    "dips_begin": sla_acc_dips_begin,
                    "dips_full": sla_acc_dips_full,
                    "dips_partial": sla_acc_dips_partial,
                }

                data["sla"] = {
                    "vanilla": artifacts["vanilla"]["data"],
                    "dips_begin": artifacts["begin"]["data"],
                    "dips_full": artifacts["full1"]["data"],
                    "dips_partial": artifacts["partial"]["data"],
                }
                models["sla"] = {
                    "vanilla": artifacts["vanilla"]["models"],
                    "dips_begin": artifacts["begin"]["models"],
                    "dips_full": artifacts["full1"]["models"],
                    "dips_partial": artifacts["partial"]["models"],
                }

            if "FlexMatch" in algorithm_list:
                print("Running Flex match...")
                (
                    flex_acc_vanilla,
                    flex_acc_dips_begin,
                    flex_acc_dips_full,
                    flex_acc_dips_partial,
                    artifacts,
                ) = run_FlexMatch(
                    x_unlabeled=x_unlabeled,
                    x_test=x_test,
                    y_test=y_test,
                    x_train=x_train,
                    y_train=y_train,
                    upper_threshold=upper_threshold,
                    numIters=numIters,
                    nest=nest,
                    seed=seed,
                    easy_train=easy_train,
                    dips_metric=dips_metric,
                    dips_xthresh=dips_xthresh,
                    dips_ythresh=dips_ythresh,
                    verbose=verbose,
                )

                results["flex"] = {
                    "vanilla": flex_acc_vanilla,
                    "dips_begin": flex_acc_dips_begin,
                    "dips_full": flex_acc_dips_full,
                    "dips_partial": flex_acc_dips_partial,
                }

                data["flex"] = {
                    "vanilla": artifacts["vanilla"]["data"],
                    "dips_begin": artifacts["begin"]["data"],
                    "dips_full": artifacts["full1"]["data"],
                    "dips_partial": artifacts["partial"]["data"],
                }
                models["flex"] = {
                    "vanilla": artifacts["vanilla"]["models"],
                    "dips_begin": artifacts["begin"]["models"],
                    "dips_full": artifacts["full1"]["models"],
                    "dips_partial": artifacts["partial"]["models"],
                }

            overall_result_dicts.append(results)
            overall_data_dicts.append(data)
            overall_model_dicts.append(models)

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(e)
            pass

    return overall_result_dicts, overall_data_dicts, overall_model_dicts, datasize
