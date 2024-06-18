model="wideresnet"
num_labeled=1000
batch_size=16
lr=0.03
super_epochs=50
noise_type="worse_label"
seed=0
percentile_begin=20 #percentage of samples to be cleaned initially
percentile_iterative=10 #percentage of samples to be cleaned iteratively during the semi supervised iterations



python ../train.py --dataset cifar10 --num-labeled $num_labeled --arch $model --batch-size $batch_size --lr $lr --seed $seed --out ../results/cifar10@4000.5 --use-dips --iterative-dips --super-epochs $super_epochs --noise-type $noise_type --percentile-cleaning $percentile_begin --percentile-iterative $percentile_iterative
 