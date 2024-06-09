# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay


def iteration_evolution_plot(results):
    n_samples = len(results["pseudo"]["vanilla_mean"])

    plt.figure(figsize=(8, 5))

    # Supervised Learning
    supervised_learning_result = [
        results["supervised_learning_accuracy"]["acc_mean"]
    ] * n_samples

    plt.plot(
        np.arange(n_samples),
        supervised_learning_result,
        "m:",
        linewidth=4,
        label="Supervised Learning",
    )

    # Supervised Learning - preprocessing
    supervised_learning_result_easy = [
        results["supervised_learning_accuracy_easy"]["acc_mean"]
    ] * n_samples

    plt.plot(
        np.arange(n_samples),
        supervised_learning_result_easy,
        "m:",
        linewidth=4,
        label="Preprocess + Supervised Learning",
    )

    # Pseudo Labeling
    plt.plot(
        results["pseudo"]["vanilla_mean"], "k-.", linewidth=4, label="Pseudo-labeling"
    )
    plt.plot(
        results["pseudo"]["diq_begin_mean"], linewidth=4, label="Pseudo-labeling (DIQ)"
    )
    plt.plot(
        results["pseudo"]["diq_full_mean"],
        linewidth=4,
        label="Pseudo-labeling (DIQ-FULL)",
    )
    plt.plot(
        results["pseudo"]["diq_full2_mean"],
        linewidth=4,
        label="Pseudo-labeling (DIQ-FULL2)",
    )

    # UPS
    plt.plot(results["ups"]["vanilla_mean"], "k-.", linewidth=4, label="UPS")
    plt.plot(results["ups"]["diq_begin_mean"], linewidth=4, label="UPS (DIQ)")
    plt.plot(results["ups"]["diq_full_mean"], linewidth=4, label="UPS (DIQ-FULL)")
    plt.plot(results["ups"]["diq_full2_mean"], linewidth=4, label="UPS (DIQ-FULL2)")

    # SLA
    plt.plot(results["sla"]["vanilla_mean"], "k-.", linewidth=4, label="SLA")
    plt.plot(results["sla"]["diq_begin_mean"], linewidth=4, label="SLA (DIQ)")
    plt.plot(results["sla"]["diq_full_mean"], linewidth=4, label="SLA (DIQ-FULL)")
    plt.plot(results["sla"]["diq_full2_mean"], linewidth=4, label="SLA (DIQ-FULL2)")

    # CSA
    plt.plot(results["csa"]["vanilla_mean"], "k-.", linewidth=4, label="CSA")
    plt.plot(results["csa"]["diq_begin_mean"], linewidth=4, label="CSA (DIQ)")
    plt.plot(results["csa"]["diq_full_mean"], linewidth=4, label="CSA (DIQ-FULL)")
    plt.plot(results["csa"]["diq_full2_mean"], linewidth=4, label="CSA (DIQ-FULL2)")

    # FlexMatch
    plt.plot(results["flex"]["vanilla_mean"], "k-.", linewidth=4, label="FlexMatch")
    plt.plot(results["flex"]["diq_begin_mean"], linewidth=4, label="FlexMatch (DIQ)")
    plt.plot(
        results["flex"]["diq_full_mean"], linewidth=4, label="FlexMatch (DIQ-FULL)"
    )
    plt.plot(
        results["flex"]["diq_full2_mean"], linewidth=4, label="FlexMatch (DIQ-FULL2)"
    )
    plt.legend()
    plt.xlabel("Number of iterations", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.show()


def bar_plot_compare(results, dataset_name, datasize, trials, save_folder="./figures"):
    # create data
    x = 1.5 * np.arange(5)
    fully_supervised = [results["fully_supervised_learning_accuracy"]["acc_mean"]] * 5
    supervised = [results["supervised_learning_accuracy"]["acc_mean"]] * 5
    preprocess_supervised = [
        results["supervised_learning_accuracy_easy"]["acc_mean"]
    ] * 5
    baseline_ssl = [
        results["pseudo"]["vanilla_mean"],
        results["ups"]["vanilla_mean"],
        results["flex"]["vanilla_mean"],
        results["sla"]["vanilla_mean"],
        results["csa"]["vanilla_mean"],
    ]
    diq_begin = [
        results["pseudo"]["diq_begin_mean"],
        results["ups"]["diq_begin_mean"],
        results["flex"]["diq_begin_mean"],
        results["sla"]["diq_begin_mean"],
        results["csa"]["diq_begin_mean"],
    ]
    diq_full = [
        results["pseudo"]["diq_full_mean"],
        results["ups"]["diq_full_mean"],
        results["flex"]["diq_full_mean"],
        results["sla"]["diq_full_mean"],
        results["csa"]["diq_full_mean"],
    ]
    diq_full2 = [
        results["pseudo"]["diq_full2_mean"],
        results["ups"]["diq_full2_mean"],
        results["flex"]["diq_full2_mean"],
        results["sla"]["diq_full2_mean"],
        results["csa"]["diq_full2_mean"],
    ]
    width = 0.15

    # plot data in grouped manner of bar type
    plt.bar(x - 0.45, preprocess_supervised, width)
    plt.bar(x - 0.3, supervised, width)
    plt.bar(x - 0.15, baseline_ssl, width)
    plt.bar(x + 0, diq_begin, width)
    plt.bar(x + 0.15, diq_full, width)
    plt.bar(x + 0.3, diq_full2, width)
    plt.bar(x + 0.45, fully_supervised, width)
    plt.xticks(x, ["Pseudo", "UPS", "Flex", "SLA", "CSA"])
    plt.xlabel("SSL Method")
    plt.ylabel("Performance (%)")
    plt.legend(
        [
            "Preprocess + Supervised",
            "'Supervised'",
            "'Baseline'",
            "'DIQ (Begin)'",
            "'DIQ (Begin+Full)'",
            "'DIQ (Full)'",
            "'Fully Supervised'",
        ]
    )

    min = results["supervised_learning_accuracy"]["acc_mean"] - 5
    max = results["fully_supervised_learning_accuracy"]["acc_mean"] + 5

    plt.ylim(min, max)
    plt.title(f"Dataset: {dataset_name}, Data Size: {datasize}, Trials: {trials}")

    if save_folder is not None:
        # stdlib
        import os

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = f"{dataset_name}_{datasize}_{trials}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath)
        plt.show()


def adapted_boxplot(results, lower_slack=5, upper_slack=5, title=""):
    # third party
    import matplotlib.pyplot as plt
    import pandas as pd

    models = []
    for model in results.keys():
        if model not in (
            "fully_supervised_learning_accuracy",
            "supervised_learning_accuracy",
            "supervised_learning_accuracy_easy",
            "dataset",
            "x_thresh",
            "y_thresh",
            "num_trials",
            "seed",
            "prop_data",
            "prop_lab",
            "full_supervised_learning_accuracy_easy",
            "supervised_learning_accuracy",
            "supervised_learning_accuracy_easy",
            "supervised_learning_accuracy",
            "supervised_learning_accuracy_easy",
            "fully_supervised_y_scores",
            "fully_supervised_easy_y_scores",
            "supervised_y_scores",
            "supervised_easy_y_scores",
        ):
            models.append(model)

    d = {}
    plt.figure()
    for model in models:
        tmp_dict = {}
        tmp_dict["supervised"] = results["supervised_learning_accuracy"]["acc_mean"]
        tmp_dict["vanilla"] = results[model]["vanilla_mean"]
        # tmp_dict["preprocess supervised"] = results[
        #     "supervised_learning_accuracy_easy"
        # ]["acc_mean"]

        # tmp_dict["diq (begin)"] = results[model]["diq_begin_mean"]

        # tmp_dict["diq (iter)"] = results[model]["diq_full2_mean"]
        tmp_dict["diq (begin+iter)"] = results[model]["diq_full_mean"]
        # tmp_dict["fully supervised"] = results["fully_supervised_learning_accuracy"][
        #     "acc_mean"
        # ]
        # tmp_dict["preprocess supervised"] = results[
        #     "supervised_learning_accuracy_easy"
        # ]["acc_mean"]

        d[model] = tmp_dict
        min = results["supervised_learning_accuracy"]["acc_mean"] - lower_slack
        max = results["supervised_learning_accuracy"]["acc_mean"] + upper_slack
    return pd.DataFrame(d), min, max
    # .T.plot(kind="bar", ylim=(min, max), title=title)
    # plt.show()


def visualize_semi(X_unlab, X_lab, y_lab):
    """Visualize unlabeled and labeled data. The labeled data should have colors, while the unlabeled data should be grey.
    Args:
        X_unlab (_type_): _description_
        X_lab (_type_): _description_
        y_lab (_type_): _description_
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_unlab[:, 0], X_unlab[:, 1], color="grey", alpha=0.5, label="Unlabeled"
    )
    plt.scatter(X_lab[:, 0], X_lab[:, 1], c=y_lab, cmap="tab10", label="Labeled")
    plt.legend()
    plt.show()


def visualize_snapshots(
    X_lab_tilde,
    y_lab_tilde,
    X_unlab,
    indices_easy,
    indices_ambiguous,
    indices_hard,
    model,
):
    """
    Plot the data in X_lab_tilde with color y_lab_tilde as stars markers, ambiguous indices in X_lab_tilde or hard indices in X_lab_tilde should be red otherwise
    plot the X_unlab as circles in grey
    plot the decision boundary of the model
    """
    disp = DecisionBoundaryDisplay.from_estimator(
        model, X_unlab, response_method="predict", cmap="magma", alpha=0.3, eps=0.5
    )
    disp.ax_.scatter(
        X_unlab[:, 0], X_unlab[:, 1], color="grey", alpha=0.5, label="Unlabeled"
    )

    disp.ax_.scatter(
        X_lab_tilde[indices_easy, 0],
        X_lab_tilde[indices_easy, 1],
        c=y_lab_tilde[indices_easy],
        marker="*",
        cmap="tab10",
        label="Easy",
    )
    if len(indices_ambiguous) > 0:
        disp.ax_.scatter(
            X_lab_tilde[indices_ambiguous, 0],
            X_lab_tilde[indices_ambiguous, 1],
            color="red",
            marker="*",
            label="Ambiguous",
        )
    if len(indices_hard) > 0:
        disp.ax_.scatter(
            X_lab_tilde[indices_hard, 0],
            X_lab_tilde[indices_hard, 1],
            color="black",
            marker="*",
            label="Hard",
        )
    plt.legend()
    plt.show()


def visualize_ensemble_snapshots(
    artifacts, results, list_method=["vanilla", "begin", "full1", "full2"]
):
    """
    TODO: plot the figures in a grid
    """
    for method in list_method:
        data = artifacts[method]["data"]
        models = artifacts[method]["models"]
        indices = artifacts[method]["indices"]
        test_accuracy = results[method]

        for i in range(len(models)):
            (
                X_lab_tilde,
                y_lab_tilde,
                X_unlab,
                indices_easy,
                indices_ambiguous,
                indices_hard,
                model,
            ) = (
                data[i]["X"],
                data[i]["y"],
                data[i]["X_unlab"],
                indices[i][0],
                indices[i][1],
                indices[i][2],
                models[i],
            )
            disp = DecisionBoundaryDisplay.from_estimator(
                model,
                X_unlab,
                response_method="predict",
                cmap="magma",
                alpha=0.3,
                eps=0.5,
            )
            disp.ax_.scatter(
                X_unlab[:, 0], X_unlab[:, 1], color="grey", alpha=0.5, label="Unlabeled"
            )

            disp.ax_.scatter(
                X_lab_tilde[indices_easy, 0],
                X_lab_tilde[indices_easy, 1],
                c=y_lab_tilde[indices_easy],
                marker="*",
                cmap="tab10",
                label="Easy",
            )
            if len(indices_ambiguous) > 0:
                disp.ax_.scatter(
                    X_lab_tilde[indices_ambiguous, 0],
                    X_lab_tilde[indices_ambiguous, 1],
                    color="red",
                    marker="*",
                    label="Ambiguous",
                )
            if len(indices_hard) > 0:
                disp.ax_.scatter(
                    X_lab_tilde[indices_hard, 0],
                    X_lab_tilde[indices_hard, 1],
                    color="black",
                    marker="*",
                    label="Hard",
                )
            plt.legend()
