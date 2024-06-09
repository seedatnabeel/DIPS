#!/bin/bash

# Define variables
numTrials=1
# if set to zero we then go adaptive
dips_xthresh=0
seed=0
project_name="test_efficiency"
method='dips'


##### minimum runs from a compute efficiency perspective
echo "compas"
python ../src/run_data_efficiency_sweep.py --dataset-name compas --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

echo "covid"
python ../src/run_data_efficiency_sweep.py --dataset-name covid --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

echo "adult"
python ../src/run_data_efficiency_sweep.py --dataset-name adult --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

echo "maggic"
python ../src/run_data_efficiency_sweep.py --dataset-name maggic --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

###### full datasets
# echo "seer"
# python ../src/run_data_efficiency_sweep.py --dataset-name seer --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# echo "cutract"
# python ../src/run_data_efficiency_sweep.py --dataset-name cutract --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# echo "German-credit"
# python ../src/run_data_efficiency_sweep.py --dataset-name German-credit --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# echo "credit"
# python ../src/run_data_efficiency_sweep.py --dataset-name credit --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters 5 --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# #agaricus-lepiota
# python ../src/run_data_efficiency_sweep.py --dataset-name agaricus-lepiota --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# #higgs
# python ../src/run_data_efficiency_sweep.py --dataset-name higgs --seed $seed --nest 100 --prop-data 0.1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# #drug
# python ../src/run_data_efficiency_sweep.py --dataset-name drug --seed $seed --nest 100 --prop-data 0.1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

# #blog
# python ../src/run_data_efficiency_sweep.py --dataset-name blog --seed $seed --nest 100 --prop-data 0.2 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name --method $method

