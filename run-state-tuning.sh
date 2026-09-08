#!/bin/bash
#######################################################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
# !!! If launch gets stuck, clean lock files in your TORCH_EXTENSIONS_DIR !!!
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="32"
N_EMBD="4096"
#
CTX_LEN="4096" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="/mnt/nvme0n1/RWKV-LM-V7_alic/XML_tool_use_state_tuning_no_system" # set output folder
#
# !!! by default train.py will load the last .pth in PROJ_DIR, and continue training from it !!!
# !!! so here we will REMOVE all previous checkpts in PROJ_DIR, so they won't be loaded !!!
# !!! comment these if you don't want this behavior !!!
#
# rm "$PROJ_DIR"/rwkv-*0.pth
# rm "$PROJ_DIR"/rwkv-71.pth
# rm "$PROJ_DIR"/rwkv-final.pth
#
#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
# State tuning example (supply a base checkpoint with --load_model):
#   --train_type state --chunk_ctx 1024 --state_detach 0 --grad_cp 1 --head_chunk 4096
#   --lr_init 1e-5 --lr_final 1e-5
# State checkpoints contain only blocks.*.att.time_state; resume them with --load_state.
#
M_BSZ="16" # takes ~7G VRAM => reduce this to save VRAM, increase this for faster speed; try larger bsz (can use HEAD_CHUNK to save VRAM) for lower loss
LR_INIT="1e-5"
LR_FINAL="1e-5"
GRAD_CP=1 # 1 => slower, save VRAM; 0 => faster, more VRAM
HEAD_CHUNK=0 # state tuning: avoids materializing a full [B, chunk_ctx, vocab] head activation
KERNEL="@rwkv3" # "" => default; "@rwkv3" => usually faster, especially for H100
EPOCH_SAVE=1 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
#
# DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage) UPDATE: very buggy in new deepspeed, so I disabled it
#
python train.py --load_model "/mnt/nvme0n1/rwkv7-g1j-7.2b-20260831-ctx16384.pth" --wandb "" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --train_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "/mnt/nvme0n1/RWKV-LM-V7_alic/data/rwkv_qwen36_tool_call_no_system" --my_exit_tokens 21687577 --magic_prime 5279 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --kernel $KERNEL \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size 64 --head_chunk $HEAD_CHUNK --train_type state \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True #--ds_bucket_mb $DS_BUCKET_MB
