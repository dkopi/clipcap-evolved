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
# module load java/jdk-19

python <<EOF
import torch
print(f"cuda: {torch.cuda.is_available()}")
EOF

# rm -rf $TMPDIR/data
mkdir -p $TMPDIR/data/nocaps
mkdir -p $TMPDIR/pl_checkpoints/$SLURM_JOB_ID

echo "unzipping data..."

# ./data/unzip -q -n data/annotations_trainval2014.zip -d $TMPDIR/data
# ./data/unzip -q -n data/val2014.zip -d $TMPDIR/data
# ./data/unzip -q -n data/train2014.zip -d $TMPDIR/data
./data/unzip -q -n data/annotations_trainval2017.zip -d $TMPDIR/data
./data/unzip -q -n data/val2017.zip -d $TMPDIR/data
./data/unzip -q -n data/train2017.zip -d $TMPDIR/data

python generate_nocaps.py --root_dir $TMPDIR/data/nocaps

echo "starting training..."

shared_part="srun python train.py --annotations_file $TMPDIR/data/annotations/captions_train2017.json --data_dir $TMPDIR/data/train2017 --val_annotations_file $TMPDIR/data/annotations/captions_val2017.json --val_data_dir $TMPDIR/data/val2017 --checkpoint_path $TMPDIR/pl_checkpoints/$SLURM_JOB_ID --nocaps_root $TMPDIR/data/nocaps"

# $shared_part --batch_size 8 --lr 2e-5 --direct --run_name gpt_direct
# $shared_part --batch_size 8 --lr 2e-5 --lora --direct --run_name gpt_direct_lora

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --direct --arch flan-t5 --run_name flant5_direct
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --grad_clip 2.0 --direct --finetune_lm --arch flan-t5 --run_name flant5_finetuned_direct_gc2

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --run_name mlp_small
# $shared_part --mlp_hidden_size 256 --batch_size 40 --warmup 1 --run_name mlp_small_bs40
# $shared_part --mlp_hidden_size 3840 --batch_size 40 --no_cosine --warmup 3000 --finetune_lm --warmup_use_steps --run_name mlp_small_nc_bs40_3840_ft
# $shared_part --mlp_hidden_size 256 --batch_size 40 --no_cosine --warmup 3000 --activation tanh --finetune_lm --warmup_use_steps --run_name mlp_small_nc_bs40_ft
# $shared_part --mlp_hidden_size 256 --batch_size 40 --no_cosine --warmup 3000 --activation tanh --warmup_use_steps --arch clipcap --run_name clipcap_small_nc_bs40
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --warmup 1 --direct_proj --run_name mlp_small_proj
# $shared_part --mlp_hidden_size 512 --batch_size 8 --lr 2e-5 --lora --direct_proj --run_name mlp_small_proj_lora_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --direct_proj --run_name mlp_small_proj_nw

# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-4 --grad_clip 100.0 --warmup 1 --flan_size small --arch flan-t5 --run_name flant5_small_gc10
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

