#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=00:14:30
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

rm -rf checkpoints

source $HOME/.bashrc
module load cuda11.7/toolkit/11.7

python <<EOF
import torch
print(f"cuda: {torch.cuda.is_available()}")
EOF

# rm -rf $TMPDIR/data
mkdir -p $TMPDIR/data

echo "unzipping data..."

./data/unzip -q -n data/annotations_trainval2014.zip -d $TMPDIR/data
./data/unzip -q -n data/val2014.zip -d $TMPDIR/data
./data/unzip -q -n data/train2014.zip -d $TMPDIR/data

echo "starting training..."

# srun python train.py --annotations_file $TMPDIR/data/annotations/captions_train2014.json --data_dir $TMPDIR/data/train2014 --val_annotations_file $TMPDIR/data/annotations/captions_val2014.json --val_data_dir $TMPDIR/data/val2014 --mlp_hidden_size 256 --batch_size 8 --warmup 1 --use_unpooled_output --run_name rich
srun python train.py --annotations_file $TMPDIR/data/annotations/captions_train2014.json --data_dir $TMPDIR/data/train2014 --val_annotations_file $TMPDIR/data/annotations/captions_val2014.json --val_data_dir $TMPDIR/data/val2014 --mlp_hidden_size 256 --batch_size 8 --warmup 1 --arch clipcap --run_name clipcap_transformer


