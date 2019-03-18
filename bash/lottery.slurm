#!/bin/sh
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=pottery.out
#SBATCH --job-name=proon
#SBATCH --gres=gpu:2
export PATH="$HOME/miniconda/bin:$PATH"

source activate bertie
echo 'bertie activated'
nvidia-smi

cd ..

python train.py --net='resnet18' --data_loc='../prunes/data' --save_file='resnet18' --lottery --reset_epoch=2
python prune.py --net='resnet18' --data_loc='../prunes/data' --save_file='resnet18_fisher' --base_model='resnet18'

# compare lottery tickets to random reinit
for k in {769744,3497477,6225211,8952944,11680678};
do
    python train.py --GPU 0 --net='resnet18' --data_loc='../prunes/data' --deploy --mask=1 --base_file="resnet18_fisher_1900_prunes" --params_left $k --save_file="rnet18_scratch_"$k"_params"
    python train.py --GPU 1 --net='resnet18' --data_loc='../prunes/data' --deploy --mask=1 --base_file="lticket_resnet18_fisher_1900_prunes" --params_left 4k --save_file="rnet18_ltick_"$k"_params"
done

source deactivate
