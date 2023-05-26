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


## proj+decoder - different mlp sizes

## $shared_part --mlp_hidden_size 32 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_64
## $shared_part --mlp_hidden_size 128 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_128
## $shared_part --mlp_hidden_size 256 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_256
## $shared_part --mlp_hidden_size 512 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_512
## $shared_part --mlp_hidden_size 512 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_1024
## $shared_part --mlp_hidden_size 2048 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj_2048



## mlp+flant5

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --arch flan-t5 --flan_size small --run_name flant5_small_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --arch flan-t5 --flan_size base --run_name flant5_base_ft


## proj+flant5

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --direct_proj --arch flan-t5 --flan_size small --run_name flant5_proj_small_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --direct_proj --arch flan-t5 --flan_size base --run_name flant5_proj_base_ft


## proj+decoder

## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_proj
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --finetune_lm --run_name flan_mlp_small_proj_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_proj
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --finetune_lm --run_name flan_mlp_base_proj_ft
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_proj


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



# ================== REEVALS ========================


## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --t5 --eval_ckpt_path /tmp/pl_checkpoints/66601/epoch=9-step=141680.ckpt --run_name t5_mlp_base_proj #node009
## $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --t5 --finetune_lm --eval_ckpt_path /tmp/pl_checkpoints/66602/epoch=9-step=141680.ckpt --run_name t5_mlp_base_proj_ft #node007

## $shared_part --lr 2e-5 --arch clipcap --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --prefix_length 40 --eval_ckpt_path /tmp/pl_checkpoints/66361/epoch=3-step=56672.ckpt --run_name clipcap_trans #node011



# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --eval_ckpt_path /tmp/pl_checkpoints/66362/epoch=4-step=70840.ckpt --run_name clipcap_mlp #node012
# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --eval_ckpt_path /tmp/pl_checkpoints/66363/epoch=5-step=85008.ckpt --finetune_lm --run_name clipcap_mlp_ft #node013

# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --finetune_lm --arch flan-t5 --flan_size base --eval_ckpt_path /tmp/pl_checkpoints/66360/epoch=9-step=141680.ckpt --run_name flant5_base_ft #node009

# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --finetune_lm --eval_ckpt_path /tmp/pl_checkpoints/66369/epoch=9-step=141680.ckpt --run_name flan_mlp_base_proj_ft #node007

# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --finetune_lm --eval_ckpt_path /tmp/pl_checkpoints/66367/epoch=9-step=141680.ckpt --run_name flan_mlp_small_proj_ft #node026

# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size small --direct_proj --arch flan-mlp --eval_ckpt_path /tmp/pl_checkpoints/66366/epoch=8-step=127512.ckpt --run_name flan_mlp_small_proj #node025

# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size base --direct_proj --arch flan-mlp --eval_ckpt_path /tmp/pl_checkpoints/66368/epoch=5-step=85008.ckpt --run_name flan_mlp_base_proj #node006

# $shared_part --mlp_hidden_size 3840 --lr 2e-5 --batch_size 40 --no_cosine --warmup 5000 --warmup_use_steps --flan_size large --direct_proj --arch flan-mlp --eval_ckpt_path /tmp/pl_checkpoints/66399/epoch=7-step=113344.ckpt --run_name flan_mlp_large_proj #node012




# ===================================================





# $shared_part --batch_size 8 --lr 2e-5 --direct --run_name gpt_direct
# $shared_part --batch_size 8 --lr 2e-5 --lora --direct --run_name gpt_direct_lora

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --direct --arch flan-t5 --run_name flant5_direct
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --grad_clip 2.0 --direct --finetune_lm --arch flan-t5 --run_name flant5_finetuned_direct_gc2

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --run_name mlp_small
# $shared_part --mlp_hidden_size 256 --batch_size 40 --warmup 1 --run_name mlp_small_bs40
# $shared_part --mlp_hidden_size 3840 --batch_size 40 --no_cosine --warmup 3000 --activation tanh --finetune_lm --warmup_use_steps --run_name mlp_small_nc_bs40_3840_ft_tanh
# $shared_part --mlp_hidden_size 256 --batch_size 40 --no_cosine --warmup 3000 --activation tanh --finetune_lm --warmup_use_steps --run_name mlp_small_nc_bs40_ft
# $shared_part --mlp_hidden_size 256 --batch_size 40 --no_cosine --warmup 3000 --activation tanh --warmup_use_steps --epochs 1 --arch clipcap --run_name clipcap_small_nc_bs40
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --warmup 1 --direct_proj --run_name mlp_small_proj
# $shared_part --mlp_hidden_size 512 --batch_size 8 --lr 2e-5 --lora --direct_proj --run_name mlp_small_proj_lora_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --direct_proj --run_name mlp_small_proj_nw

# $shared_part --mlp_hidden_size 3840 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --warmup 1 --finetune_lm --flan_size small --arch flan-t5 --run_name flant5_small_gc100_ft_3840
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --warmup 1 --flan_size base --arch flan-t5 --run_name flant5_base_gc10
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --warmup 1 --flan_size base --arch flan-t5 --direct_proj --run_name flant5_base__direct_proj_gc100
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size small --finetune_lm --arch flan-t5 --run_name flant5_small_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --arch flan-t5 --run_name flant5_nw_gc10
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --finetune_lm --arch flan-t5 --run_name flant5_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --finetune_lm --arch flan-mlp --run_name flan_mlp_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size small --arch flan-mlp --run_name flan_mlp_small_nw_gc10
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --finetune_lm --arch flan-transformer --run_name flan_tr_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_direct_proj_nw_gc10_ft

# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size small --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_small_direct_proj_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 10.0 --warmup 1 --flan_size small --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc10_ft_2e4
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 10.0 --flan_size small --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc10_ft_2e4_nw
# $shared_part --mlp_hidden_size 2048 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --flan_size small --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc100_ft_2e4_nw_2048
# $shared_part --mlp_hidden_size 2048 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc100_2e4_nw_2048
# $shared_part --mlp_hidden_size 2048 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_nw_2048
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-3 --grad_clip 10.0 --flan_size small --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc10_ft_2e3_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-3 --grad_clip 10.0 --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc100_2e3_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-3 --grad_clip 10.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e3_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_nw

# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --val_freq 250 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_nw_bs64
# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --val_freq 250 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_bs64
# $shared_part --mlp_hidden_size 256 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_nw
# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4
# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_ds2017
# $shared_part --mlp_hidden_size 3840 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_ds2017_tanh_3840
# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_direct_proj_gc100_2e4_ds2017_tanh
# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_direct_proj_gc100_2e4_ds2017
# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_direct_proj_gc100_2e4
# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_direct_proj_gc100_2e4_ds2017_leaky
# $shared_part --mlp_hidden_size 3840 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size small --finetune_lm --direct_proj --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc100_2e4_ds2017_3840_ft
# $shared_part --mlp_hidden_size 3840 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size small --finetune_lm --direct_proj --arch flan-mlp --mlp_dropout 0.5 --run_name flan_mlp_small_direct_proj_gc100_2e4_ds2017_3840_ft_d05
# $shared_part --mlp_hidden_size 3840 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --finetune_lm --direct_proj --arch flan-mlp --mlp_dropout 0.5 --run_name flan_mlp_base_direct_proj_gc100_2e4_ds2017_3840_ft_d05
# $shared_part --mlp_hidden_size 512 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size small --finetune_lm --direct_proj --arch flan-mlp --mlp_dropout 0.5 --run_name flan_mlp_small_direct_proj_gc100_2e4_ds2017_512_ft_d05
# $shared_part --mlp_hidden_size 3840 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size small --lora --direct_proj --arch flan-mlp --mlp_dropout 0.5 --run_name flan_mlp_small_direct_proj_gc100_2e4_ds2017_3840_lora_d05
# $shared_part --mlp_hidden_size 3840 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --finetune_lm --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_ds2017_3840_ft

# $shared_part --mlp_hidden_size 256 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc100_2e4
# $shared_part --mlp_hidden_size 256 --batch_size 32 --eval_batches 16 --val_freq 500 --epochs 50 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_direct_proj_gc100_2e4
# $shared_part --mlp_hidden_size 256 --batch_size 32 --eval_batches 16 --val_freq 500 --epochs 50 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_2e4_bs32
# $shared_part --mlp_hidden_size 256 --batch_size 32 --eval_batches 16 --val_freq 500 --epochs 50 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_direct_proj_gc100_2e4_bs32
# $shared_part --mlp_hidden_size 1024 --batch_size 8 --eval_batches 128 --val_freq 8000 --epochs 50 --warmup 1 --lr 2e-4 --grad_clip 100.0 --flan_size large --direct_proj --arch flan-mlp --run_name flan_mlp_large_direct_proj_gc100_2e4_bs8_1024

# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --val_freq 250 --lr 2e-4 --grad_clip 100.0 --flan_size base --arch flan-transformer --run_name flan_transformer_base_gc100_2e4_nw_bs64
# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --val_freq 250 --lr 2e-4 --grad_clip 100.0 --flan_size base --arch flan-transformer --direct_proj --run_name flan_transformer_direct_proj_base_gc100_2e4_nw_bs64
# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --warmup 1 --val_freq 250 --lr 2e-4 --grad_clip 100.0 --flan_size base --arch flan-transformer --direct_proj --run_name flan_transformer_lin_direct_proj_base_gc100_2e4_bs64
# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --warmup 1 --val_freq 250 --lr 2e-4 --grad_clip 100.0 --flan_size base --arch flan-t5 --run_name flant5_base_gc100_2e4
# $shared_part --mlp_hidden_size 256 --batch_size 64 --eval_batches 4 --val_freq 250 --warmup 1 --lr 2e-3 --grad_clip 100.0 --flan_size small --arch flan-mlp --finetune_lm --direct_proj --run_name flan_mlp_direct_proj_small_gc100_2e3_bs64_ft


# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 10.0 --warmup 1 --flan_size base --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc10_ft_2e4
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-3 --grad_clip 100.0 --flan_size base --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_base_direct_proj_gc100_ft_nw_2e3
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-3 --grad_clip 100.0 --flan_size base --finetune_lm --arch flan-mlp --run_name flan_mlp_base_gc100_ft_nw_2e3
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-3 --grad_clip 100.0 --flan_size small --arch flan-mlp --run_name flan_mlp_small_gc100_nw_2e3
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_direct_proj_nw_gc10
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --direct_proj --arch flan-mlp --run_name flan_mlp_direct_proj_nw_gc10
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size small --direct_proj --arch flan-mlp --run_name flan_mlp_small_direct_proj_nw_gc10
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --direct --arch flan-mlp --finetune_lm --run_name flan_mlp_direct_nw_gc10_ft

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --finetune_lm --run_name mlp_small_finetuned
# $shared_part --mlp_hidden_size 4096 --batch_size 8 --warmup 1 --run_name mlp_big
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --use_unpooled_output --run_name rich
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --arch clipcap --run_name clipcap_transformer
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --use_unpooled_output --arch clipcap --run_name clipcap_t_rich

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --lora --gpt_size medium --run_name gpt_medium_lora
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --lora --gpt_size large --run_name gpt_large_lora


# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --grad_clip 2.0 --arch flan-t5 --run_name flant5_gc2
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --grad_clip 10.0 --finetune_lm --arch flan-t5 --run_name flant5_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --finetune_lm --arch flan-t5 --run_name flant5_gc10_ft_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --finetune_lm --arch flan-t5 --run_name flant5_gc10_ft_nw_ogloss
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --finetune_lm --arch flan-t5 --flan_size base --run_name flant5_base_gc10_ft_nw_ogloss
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --finetune_lm --arch flan-mlp --flan_size base --run_name flant_mlp_base_gc10_ft_nw_ogloss
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --finetune_lm --arch flan-mlp --flan_size base --run_name t5_mlp_base_gc10_ft_nw_ogloss

# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 2.0 --arch flan-transformer --flan_size small --run_name flant5_transformer_nw_gc2
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 2.0 --finetune_lm --arch flan-t5 --flan_size small --run_name flant5_finetunedencoder_nw_gc2

# $shared_part --batch_size 8 --warmup 1 --lr 2e-5 --direct --finetune_lm --arch flan-mlp --flan_size base --run_name flan_decoder_direct
# $shared_part --batch_size 8 --lr 2e-5 --direct --grad_clip 2.0 --finetune_lm --arch flan-mlp --flan_size base --run_name flan_decoder_direct_gc2_nowarmup

