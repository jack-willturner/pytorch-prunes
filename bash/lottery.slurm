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

python train.py --net='resnet18' --data_loc='../prunes/data' --save_file='resnet18_1' --lottery --reset_epoch=2
python prune.py --GPU 0 --net='resnet18' --data_loc='../prunes/data' --save_file='resnet18_fisher' --base_model='resnet18' &
python prune.py --GPU 1 --net='resnet18' --data_loc='../prunes/data' --save_file='resnet18_l1' --base_model='resnet18' --l1_prune=True

echo "pruning done"

# compare lottery tickets to random reinit
for k in 300 600 900 1200 1500 1800
do
    python train.py --GPU 0 --net='resnet18' --data_loc='../prunes/data' --deploy --mask=1 --base_file="resnet18_fisher_"$k"_prunes" --save_file="r18_fisher_scratch_"$k"_prunes" &
    python train.py --GPU 1 --net='resnet18' --data_loc='../prunes/data' --deploy --mask=1 --base_file="resnet18_fisher_"$k"_prunes" --lottery --lottery_file="lticket_resnet18.t7" --save_file="r18_fisher_ltick_"$k"_prunes"
done


echo "fisher scratch done"

# compare lottery tickets to random reinit
for k in 300 600 900 1200 1500 1800
do
    python train.py --GPU 0 --net='resnet18' --data_loc='../prunes/data' --deploy --mask=1 --base_file="resnet18_l1_"$k"_prunes" --save_file="r18_l1_scratch_"$k"_prunes" &
    python train.py --GPU 1 --net='resnet18' --data_loc='../prunes/data' --deploy --mask=1 --base_file="lticket_resnet18_l1_"$k"_prunes" --lottery --lottery_file="lticket_resnet18.t7" --save_file="r18_l1_ltick_"$k"_prunes"
done



source deactivate
