#!/usr/bin/env python

# import sys
# sys.path.append('..')

import os
import argparse
import pickle
import tempfile

from method_sweep import run_baselines
from data_loader import *

import yaml
import wandb


def main():
    parser = argparse.ArgumentParser(description="Run baselines on a dataset.")
    parser.add_argument(
        "--dataset-name", type=str, default="seer", help="Name of the dataset"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--nest", type=int, default=100, help="Number of estimators for XGBoost"
    )
    parser.add_argument(
        "--prop-data", type=float, default=0.1, help="Proportion of data to use"
    )
    parser.add_argument(
        "--num-XGB-models",
        type=int,
        default=2,
        help="Number of XGBoost models to train",
    )
    parser.add_argument(
        "--numTrials", type=int, default=5, help="Number of trials to run"
    )
    parser.add_argument(
        "--numIters", type=int, default=5, help="Number of iterations to run"
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=0.8,
        help="Upper threshold for Pseudo-Labeling",
    )
    parser.add_argument(
        "--dips-metric", type=str, default="aleatoric", help="DIPS metric to use"
    )
    parser.add_argument(
        "--dips-xthresh", type=float, default=0.15, help="DIPS x threshold"
    )
    parser.add_argument(
        "--dips-ythresh", type=float, default=0.2, help="DIPS y threshold"
    )
    # projject name add argument
    parser.add_argument(
        "--project-name",
        type=str,
        default="test_project_ssl_dcai",
        help="Name of the project",
    )
    parser.add_argument("--method", type=str, default="DIPS", help="selector")

    args = parser.parse_args()

    # Load the WANDB YAML file
    with open("../wandb.yaml") as file:
        wandb_data = yaml.load(file, Loader=yaml.FullLoader)

    os.environ["WANDB_API_KEY"] = wandb_data["wandb_key"]
    wandb_entity = wandb_data["wandb_entity"]

    wandb.init(
        project=str(args.project_name),
        entity=wandb_entity,
    )

    arg_dict = vars(args)
    wandb.log(arg_dict)

    algorithm_list = [
        "Supervised_Learning",
        "Pseudo_Labeling",
        "FlexMatch",
        "UPS",
        "SLA",
        "CSA",
    ]

    # algorithm_list = [
    #     "Supervised_Learning",
    #     "Pseudo_Labeling",
    # ]

    (overall_result_dicts, overall_data_dicts, overall_model_dicts, datasize) = (
        run_baselines(
            numTrials=args.numTrials,
            numIters=args.numIters,
            upper_threshold=args.upper_threshold,
            num_XGB_models=args.num_XGB_models,
            nest=args.nest,
            seed=args.seed,
            dataset_name=args.dataset_name,
            dips_metric=args.dips_metric,
            dips_xthresh=args.dips_xthresh,
            dips_ythresh=args.dips_ythresh,
            prop_data=args.prop_data,
            verbose=False,
            algorithm_list=algorithm_list,
            method=args.method,
            epochs=20,
        )
    )

    metainfo = f"{args.dataset_name}_{args.prop_data}_{args.dips_xthresh}_{args.numTrials}_{args.seed}"

    # log overall_result_dicts to wandb as a pickle
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(overall_result_dicts, temp_file)
        temp_file_path = temp_file.name

    # Log the pickle as a wandb artifact
    artifact = wandb.Artifact(f"results_dict_{metainfo}", type="pickle")
    artifact.add_file(temp_file_path, name=f"results_dict_{metainfo}.pkl")
    wandb.run.log_artifact(artifact)
    # Clean up the temporary file
    os.remove(temp_file_path)

    # commented out to save logging space, but can be logged as well

    # # log overall_data_dicts to wandb as a pickle
    # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #     pickle.dump(overall_data_dicts, temp_file)
    #     temp_file_path = temp_file.name

    # # Log the pickle as a wandb artifact
    # artifact = wandb.Artifact(f"data_dict_{metainfo}", type="pickle")
    # artifact.add_file(temp_file_path, name=f"data_dict_{metainfo}.pkl")
    # wandb.run.log_artifact(artifact)
    # # Clean up the temporary file
    # os.remove(temp_file_path)

    # log overall_model_dicts to wandb as a pickle
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(overall_model_dicts, temp_file)
        temp_file_path = temp_file.name

    # Log the pickle as a wandb artifact
    artifact = wandb.Artifact(f"model_dict_{metainfo}", type="pickle")
    artifact.add_file(temp_file_path, name=f"model_dict_{metainfo}.pkl")
    wandb.run.log_artifact(artifact)
    # Clean up the temporary file
    os.remove(temp_file_path)

    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    main()
