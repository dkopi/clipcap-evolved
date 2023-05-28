#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90

hostname

rm -rf checkpoints

source $HOME/.bashrc
module load cuda11.7/toolkit/11.7
module load java/jdk-8

python <<EOF
import torch
print(f"cuda: {torch.cuda.is_available()}")
EOF

# rm -rf $TMPDIR/data
mkdir -p $TMPDIR/data/nocaps
mkdir -p $TMPDIR/pl_checkpoints/$SLURM_JOB_ID

echo "unzipping data..."

# ./data/unzip -q -n data/annotations_trainval2014.zip -d $TMPDIR/data
cp -n data/train.json $TMPDIR/data/train.json
cp -n data/val.json $TMPDIR/data/val.json
cp -n data/test.json $TMPDIR/data/test.json
./data/unzip -q -n data/val2014.zip -d $TMPDIR/data
./data/unzip -q -n data/train2014.zip -d $TMPDIR/data
# ./data/unzip -q -n data/annotations_trainval2017.zip -d $TMPDIR/data
# ./data/unzip -q -n data/val2017.zip -d $TMPDIR/data
# ./data/unzip -q -n data/train2017.zip -d $TMPDIR/data

rm -rf $TMPDIR/data/nocaps
python generate_nocaps.py --root_dir $TMPDIR/data/nocaps

echo "starting training..."

shared_part="srun python train.py --annotations_file $TMPDIR/data/train.json --data_dir $TMPDIR/data --val_annotations_file $TMPDIR/data/val.json --test_annotations_file $TMPDIR/data/test.json --checkpoint_path $TMPDIR/pl_checkpoints/$SLURM_JOB_ID --nocaps_root $TMPDIR/data/nocaps"




# ===================================================

## lora

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --lora --lora_target_modules all --run_name clipcap_mlp_lora_all
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --lora --lora_target_modules c_attn c_proj --run_name clipcap_mlp_lora_part

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --lora --lora_target_modules all --run_name flan_mlp_small_proj_lora_all
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --lora --lora_target_modules q v --run_name flan_mlp_small_proj_lora_part

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --lora --lora_target_modules all --run_name flan_mlp_base_proj_lora_all
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --lora --lora_target_modules q v --run_name flan_mlp_base_proj_lora_part

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size large --direct_proj --arch flan-mlp --lora --lora_target_modules all --run_name flan_mlp_large_proj_lora_all
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size large --direct_proj --arch flan-mlp --lora --lora_target_modules q v --run_name flan_mlp_large_proj_lora_part

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps  --direct_proj --lora --lora_target_modules all --run_name clipcap_mlp_proj_lora_all
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps  --direct_proj --lora --lora_target_modules c_attn c_proj --run_name clipcap_mlp_proj_lora_all


## proj+decoder - different mlp sizes

## $shared_part --mlp_hidden_size 32 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_32
## $shared_part --mlp_hidden_size 128 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_128
## $shared_part --mlp_hidden_size 256 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_256
## $shared_part --mlp_hidden_size 512 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_512
## $shared_part --mlp_hidden_size 2048 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_2048



## mlp+flant5

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --arch flan-t5 --flan_size small --run_name flant5_small_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --arch flan-t5 --flan_size base --run_name flant5_base_ft


## proj+flant5

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --direct_proj --arch flan-t5 --flan_size small --run_name flant5_proj_small_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --direct_proj --arch flan-t5 --flan_size base --run_name flant5_proj_base_ft

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --arch flan-t5-trans --run_name flant5_trans_small
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --arch flan-t5-trans --run_name flant5_trans_base


## proj+decoder

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --finetune_lm --run_name flan_mlp_small_proj_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_proj
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --finetune_lm --run_name flan_mlp_base_proj_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_proj

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --arch flan-mlp --finetune_lm --run_name flan_mlp_small_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --arch flan-mlp --finetune_lm --run_name flan_mlp_base_ft


## proj+decoder(t5)

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --t5 --run_name t5_mlp_base_proj
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --t5 --finetune_lm --run_name t5_mlp_base_proj_ft


## trans+decoder

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --arch flan-transformer --run_name flan_trans_small
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --arch flan-transformer --run_name flan_trans_base


## baselines

## $shared_part --lr 2e-5 --arch clipcap --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --prefix_length 40 --run_name clipcap_trans
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --run_name clipcap_mlp
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --run_name clipcap_mlp_ft


## proj+gpt
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps  --direct_proj --run_name clipcap_mlp_proj
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps  --direct_proj --finetune_lm --run_name clipcap_mlp_proj_ft



