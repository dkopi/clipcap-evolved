#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --nodelist=node025
#SBATCH --ntasks-per-node=1
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --time=08:15:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

rm -rf checkpoints wandb

source $HOME/.bashrc
module load cuda11.7/toolkit/11.7

python <<EOF
import torch
print(f"cuda: {torch.cuda.is_available()}")
EOF

mkdir -p $TMPDIR/data

echo "unzipping data..."

./data/unzip -q -n data/annotations_trainval2014.zip -d $TMPDIR/data
./data/unzip -q -n data/val2014.zip -d $TMPDIR/data
./data/unzip -q -n data/train2014.zip -d $TMPDIR/data

echo "starting training..."

srun python train.py --annotations_file $TMPDIR/data/annotations/captions_train2014.json --data_dir $TMPDIR/data/train2014 --val_annotations_file $TMPDIR/data/annotations/captions_val2014.json --val_data_dir $TMPDIR/data/val2014 --mlp_hidden_size 256 --batch_size 8 --warmup 1


