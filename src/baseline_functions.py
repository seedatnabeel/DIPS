from copy import deepcopy
from random import sample

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from csa import CSA
    from flexmatch import FlexMatch
    from pseudo_labeling import Pseudo_Labeling
    from ups import UPS
except:
    from .csa import CSA
    from .flexmatch import FlexMatch
    from .pseudo_labeling import Pseudo_Labeling
    from .ups import UPS


def run_pseudo(
    x_unlabeled,
    x_test,
    y_test,
    x_train,
    y_train,
    numIters,
    upper_threshold,
    nest,
    seed,
    easy_train,
    dips_metric,
    dips_xthresh,
    dips_ythresh,
    verbose,
    method="dips",
    epochs=100,
):

    # VANILLA BASELINE

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    pseudo_labeling_model = Pseudo_Labeling(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    pseudo_labeling_model.x_lab = x_train

    pseudo_labeling_model.fit(x_train, y_train)
    (
        indices_vanilla,
        data_vanilla,
        models_vanilla,
    ) = pseudo_labeling_model.get_artifacts()

    pseudo_labeling_acc_vanilla = pseudo_labeling_model.test_acc

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    pseudo_labeling_model = Pseudo_Labeling(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    pseudo_labeling_model.x_lab = x_train

    pseudo_labeling_model.fit(x_train[easy_train, :], y_train[easy_train])

    pseudo_labeling_acc_dips_begin = pseudo_labeling_model.test_acc
    indices_begin, data_begin, models_begin = pseudo_labeling_model.get_artifacts()

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    pseudo_labeling_model = Pseudo_Labeling(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    pseudo_labeling_model.x_lab = x_train

    pseudo_labeling_model.fit(
        x_train[easy_train, :],
        y_train[easy_train],
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
        method=method,
        epochs=epochs,
    )

    pseudo_labeling_acc_dips_full = pseudo_labeling_model.test_acc
    indices_full, data_full, models_full = pseudo_labeling_model.get_artifacts()

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    pseudo_labeling_model = Pseudo_Labeling(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    pseudo_labeling_model.x_lab = x_train

    pseudo_labeling_model.fit(
        x_train,
        y_train,
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
        method=method,
        epochs=epochs,
    )

    pseudo_labeling_acc_dips_partial = pseudo_labeling_model.test_acc
    indices_partial, data_partial, models_partial = (
        pseudo_labeling_model.get_artifacts()
    )

    artifacts = {}
    artifacts["vanilla"] = {
        "indices": indices_vanilla,
        "data": data_vanilla,
        "models": models_vanilla,
    }
    artifacts["begin"] = {
        "indices": indices_begin,
        "data": data_begin,
        "models": models_begin,
    }
    artifacts["full"] = {
        "indices": indices_full,
        "data": data_full,
        "models": models_full,
    }
    artifacts["partial"] = {
        "indices": indices_partial,
        "data": data_partial,
        "models": models_partial,
    }

    return (
        pseudo_labeling_acc_vanilla,
        pseudo_labeling_acc_dips_begin,
        pseudo_labeling_acc_dips_full,
        pseudo_labeling_acc_dips_partial,
        artifacts,
    )


def run_UPS(
    x_unlabeled,
    x_test,
    y_test,
    x_train,
    y_train,
    numIters,
    num_XGB_models,
    nest,
    seed,
    easy_train,
    dips_metric,
    dips_xthresh,
    dips_ythresh,
    verbose,
    method="dips",
    epochs=100,
):

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    ups_model = UPS(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    ups_model.fit(x_train, y_train)
    indices_vanilla, data_vanilla, models_vanilla = ups_model.get_artifacts()

    ups_acc_vanilla = ups_model.test_acc

    ups_model = UPS(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    ups_model.fit(x_train[easy_train, :], y_train[easy_train])
    indices_begin, data_begin, models_begin = ups_model.get_artifacts()

    ups_acc_dips_begin = ups_model.test_acc

    ups_model = UPS(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    ups_model.fit(
        x_train[easy_train, :],
        y_train[easy_train],
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
        method=method,
        epochs=epochs,
    )
    indices_full, data_full, models_full = ups_model.get_artifacts()
    ups_acc_dips_full = ups_model.test_acc

    ups_model = UPS(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    ups_model.fit(
        x_train,
        y_train,
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
        method=method,
        epochs=epochs,
    )
    indices_partial, data_partial, models_partial = ups_model.get_artifacts()

    ups_acc_dips_partial = ups_model.test_acc
    artifacts = {}
    artifacts["vanilla"] = {
        "indices": indices_vanilla,
        "data": data_vanilla,
        "models": models_vanilla,
    }
    artifacts["begin"] = {
        "indices": indices_begin,
        "data": data_begin,
        "models": models_begin,
    }
    artifacts["full"] = {
        "indices": indices_full,
        "data": data_full,
        "models": models_full,
    }
    artifacts["partial"] = {
        "indices": indices_partial,
        "data": data_partial,
        "models": models_partial,
    }

    return (
        ups_acc_vanilla,
        ups_acc_dips_begin,
        ups_acc_dips_full,
        ups_acc_dips_partial,
        artifacts,
    )


def run_FlexMatch(
    x_unlabeled,
    x_test,
    y_test,
    x_train,
    y_train,
    upper_threshold,
    numIters,
    nest,
    seed,
    easy_train,
    dips_metric,
    dips_xthresh,
    dips_ythresh,
    verbose,
    method="dips",
    epochs=100,
):

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    flex_model = FlexMatch(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    flex_model.fit(x_train, y_train)
    indices_vanilla, data_vanilla, models_vanilla = flex_model.get_artifacts()

    flex_acc_vanilla = flex_model.test_acc

    flex_model = FlexMatch(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    flex_model.fit(x_train[easy_train, :], y_train[easy_train])
    indices_begin, data_begin, models_begin = flex_model.get_artifacts()

    flex_acc_dips_begin = flex_model.test_acc

    flex_model = FlexMatch(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    flex_model.fit(
        x_train[easy_train, :],
        y_train[easy_train],
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
    )
    indices_full, data_full, models_full = flex_model.get_artifacts()

    flex_acc_dips_full = flex_model.test_acc

    flex_model = FlexMatch(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        upper_threshold=upper_threshold,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )
    flex_model.fit(
        x_train,
        y_train,
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
    )
    indices_partial, data_partial, models_partial = flex_model.get_artifacts()

    flex_acc_dips_partial = flex_model.test_acc

    artifacts = {}
    artifacts["vanilla"] = {
        "indices": indices_vanilla,
        "data": data_vanilla,
        "models": models_vanilla,
    }
    artifacts["begin"] = {
        "indices": indices_begin,
        "data": data_begin,
        "models": models_begin,
    }
    artifacts["full"] = {
        "indices": indices_full,
        "data": data_full,
        "models": models_full,
    }
    artifacts["partial"] = {
        "indices": indices_partial,
        "data": data_partial,
        "models": models_partial,
    }

    return (
        flex_acc_vanilla,
        flex_acc_dips_begin,
        flex_acc_dips_full,
        flex_acc_dips_partial,
        artifacts,
    )


def run_SLA(
    x_unlabeled,
    x_test,
    y_test,
    x_train,
    y_train,
    numIters,
    num_XGB_models,
    nest,
    seed,
    easy_train,
    dips_metric,
    dips_xthresh,
    dips_ythresh,
    verbose,
    method="dips",
    epochs=100,
):

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    sla_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice=None,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    sla_model.fit(x_train, y_train)
    indices_vanilla, data_vanilla, models_vanilla = sla_model.get_artifacts()

    sla_acc_vanilla = sla_model.test_acc

    sla_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice=None,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    sla_model.fit(x_train[easy_train, :], y_train[easy_train])
    indices_begin, data_begin, models_begin = sla_model.get_artifacts()

    sla_acc_dips_begin = sla_model.test_acc

    sla_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice=None,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    sla_model.fit(
        x_train[easy_train, :],
        y_train[easy_train],
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
    )
    indices_full, data_full, models_full = sla_model.get_artifacts()

    sla_acc_dips_full = sla_model.test_acc

    sla_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice=None,
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    sla_model.fit(
        x_train,
        y_train,
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
    )
    indices_partial, data_partial, models_partial = sla_model.get_artifacts()
    sla_acc_dips_partial = sla_model.test_acc
    artifacts = {}
    artifacts["vanilla"] = {
        "indices": indices_vanilla,
        "data": data_vanilla,
        "models": models_vanilla,
    }
    artifacts["begin"] = {
        "indices": indices_begin,
        "data": data_begin,
        "models": models_begin,
    }
    artifacts["full"] = {
        "indices": indices_full,
        "data": data_full,
        "models": models_full,
    }
    artifacts["partial"] = {
        "indices": indices_partial,
        "data": data_partial,
        "models": models_partial,
    }
    return (
        sla_acc_vanilla,
        sla_acc_dips_begin,
        sla_acc_dips_full,
        sla_acc_dips_partial,
        artifacts,
    )


def run_CSA(
    x_unlabeled,
    x_test,
    y_test,
    x_train,
    y_train,
    numIters,
    num_XGB_models,
    nest,
    seed,
    easy_train,
    dips_metric,
    dips_xthresh,
    dips_ythresh,
    verbose,
    method="dips",
    epochs=100,
):

    x_unlabeled, x_test, y_test, x_train, y_train = (
        np.asarray(x_unlabeled),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(x_train),
        np.asarray(y_train),
    )

    csa_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice="ttest",
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    csa_model.fit(x_train, y_train)
    indices_vanilla, data_vanilla, models_vanilla = csa_model.get_artifacts()

    csa_acc_vanilla = csa_model.test_acc

    csa_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice="ttest",
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    csa_model.fit(x_train[easy_train, :], y_train[easy_train])
    indices_begin, data_begin, models_begin = csa_model.get_artifacts()

    csa_acc_dips_begin = csa_model.test_acc

    csa_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice="ttest",
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    csa_model.fit(
        x_train[easy_train, :],
        y_train[easy_train],
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
    )
    indices_full, data_full, models_full = csa_model.get_artifacts()

    csa_acc_dips_full = csa_model.test_acc

    csa_model = CSA(
        x_unlabeled,
        x_test,
        y_test,
        num_iters=numIters,
        confidence_choice="ttest",
        num_XGB_models=num_XGB_models,
        verbose=verbose,
        nest=nest,
        seed=seed,
    )

    csa_model.fit(
        x_train,
        y_train,
        dips=True,
        dips_metric=dips_metric,
        dips_xthresh=dips_xthresh,
        dips_ythresh=dips_ythresh,
    )
    indices_partial, data_partial, models_partial = csa_model.get_artifacts()

    csa_acc_dips_partial = csa_model.test_acc
    artifacts = {}
    artifacts["vanilla"] = {
        "indices": indices_vanilla,
        "data": data_vanilla,
        "models": models_vanilla,
    }
    artifacts["begin"] = {
        "indices": indices_begin,
        "data": data_begin,
        "models": models_begin,
    }
    artifacts["full"] = {
        "indices": indices_full,
        "data": data_full,
        "models": models_full,
    }
    artifacts["partial"] = {
        "indices": indices_partial,
        "data": data_partial,
        "models": models_partial,
    }

    return (
        csa_acc_vanilla,
        csa_acc_dips_begin,
        csa_acc_dips_full,
        csa_acc_dips_partial,
        artifacts,
    )
