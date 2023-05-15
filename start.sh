#!/bin/bash

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
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

shared_part="srun python train.py --annotations_file $TMPDIR/data/annotations/captions_train2014.json --data_dir $TMPDIR/data/train2014 --val_annotations_file $TMPDIR/data/annotations/captions_val2014.json --val_data_dir $TMPDIR/data/val2014"

# $shared_part --batch_size 8 --lr 2e-5 --direct --run_name gpt_direct
# $shared_part --batch_size 8 --lr 2e-5 --lora --direct --run_name gpt_direct_lora

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --direct --arch flan-t5 --run_name flant5_direct
# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --grad_clip 2.0 --direct --finetune_lm --arch flan-t5 --run_name flant5_finetuned_direct_gc2

# $shared_part --mlp_hidden_size 256 --batch_size 8 --warmup 1 --run_name mlp_small
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --warmup 1 --direct_proj --run_name mlp_small_proj
# $shared_part --mlp_hidden_size 512 --batch_size 8 --lr 2e-5 --lora --direct_proj --run_name mlp_small_proj_lora_nw
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --direct_proj --run_name mlp_small_proj_nw

# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --finetune_lm --arch flan-t5 --run_name flant5_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --finetune_lm --arch flan-mlp --run_name flan_mlp_nw_gc10_ft
# $shared_part --mlp_hidden_size 256 --batch_size 8 --lr 2e-5 --grad_clip 10.0 --flan_size base --direct_proj --finetune_lm --arch flan-mlp --run_name flan_mlp_direct_proj_nw_gc10_ft

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

