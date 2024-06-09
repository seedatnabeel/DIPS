#!/bin/bash

# Define variables
numTrials=10
numIters=10
# set to zero to use adaptive threshold
dips_xthresh=0
seed=0
project_name="dips_all_test"

# create runs for the following sweeps:  
#sweeps = [('seer', '1.0'), ('adult', '0.66'), ('cutract', '1.0'), ('covid', '0.66'),   ('maggic', '0.66'), ('German-credit', '1.0'),("compas", "1.0"), ("agaricus-lepiota", '1.0'), ("bio", "1.0"), ("higgs", "0.1"),   ('drug', '0.1'), ("blog", "0.2"), ("telescope", "0.66"), ("credit", "1.0")]

# #seer
# python ../src/run_sweep.py --dataset-name seer --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #adult
# python ../src/run_sweep.py --dataset-name adult --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

#cutract
python ../src/run_sweep.py --dataset-name cutract --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

#covid
python ../src/run_sweep.py --dataset-name covid --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #maggic
# python ../src/run_sweep.py --dataset-name maggic --seed $seed --nest 100 --prop-data 0.66 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #German-credit
# python ../src/run_sweep.py --dataset-name German-credit --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #compas
# python ../src/run_sweep.py --dataset-name compas --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #agaricus-lepiota
# python ../src/run_sweep.py --dataset-name agaricus-lepiota --seed $seed --nest 100 --prop-data 1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #higgs
# python ../src/run_sweep.py --dataset-name higgs --seed $seed --nest 100 --prop-data 0.1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #drug
# python ../src/run_sweep.py --dataset-name drug --seed $seed --nest 100 --prop-data 0.1 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

# #blog
# python ../src/run_sweep.py --dataset-name blog --seed $seed --nest 100 --prop-data 0.2 --num-XGB-models 3 --numTrials $numTrials --numIters $numIters --upper-threshold 0.8 --dips-metric aleatoric --dips-xthresh $dips_xthresh --dips-ythresh 0.2 --project-name $project_name

