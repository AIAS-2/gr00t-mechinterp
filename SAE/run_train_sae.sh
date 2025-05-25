#!/bin/bash
###############################################################################
#  File:     run_train_sae.sbatch
#  Purpose:  Train + save an SAE for Eagle-2B patch tokens (the script you posted)
#            and do so in the background with Slurm / sbatch on AAU AI-LAB.
###############################################################################

############################
# 1) Slurm resources
############################
#SBATCH --job-name=train_sae                # appears in squeue
#SBATCH --gres=gpu:8                        # one A100 is sufficient for 512-unit SAE
#SBATCH --cpus-per-task=12                   # dataloader helpers
#SBATCH --mem=28G                           # system RAM
#SBATCH --time=12:00:00                     # HH:MM:SS wall-clock limit
#SBATCH --output=logs/%x-%j.out             # stdout  (%x = job-name, %j = job-ID)
#SBATCH --error=logs/%x-%j.err              # stderr
#SBATCH --mail-user=dmarus21@student.aau.dk
#SBATCH --mail-type=END,FAIL

############################
# 2) Go to your project root
############################
WORKDIR=$HOME/AIAS2/AIAS2
cd "$WORKDIR"  || { echo "Workdir not found"; exit 1; }

############################
# 3) (Optional) module loads
############################
# module load cuda/12.2  gcc/11.3  singularity/4.0

############################
# 4) Hugging-Face caches on $SCRATCH
############################
if [[ -z "$SCRATCH" ]]; then
    export SCRATCH=$HOME/scratch
fi

HF_CACHE_DIR="$SCRATCH/huggingface"
mkdir -p "$HF_CACHE_DIR"/{hub,transformers,datasets}

echo "HF caches  →  $HF_CACHE_DIR"

############################
# 5) Paths that the Python script expects
############################
IMG_DIR=$WORKDIR/openimages_subset/images_train                 # folder used inside the script
OUT_CKPT=$WORKDIR/checkpoints/sae_layer12.pt

############################
# 6) Fire up the container & run
############################
singularity exec --nv \
    --bind "$HF_CACHE_DIR:$HF_CACHE_DIR" \
    --env "HF_HOME=$HF_CACHE_DIR" \
    --env "TRANSFORMERS_CACHE=$HF_CACHE_DIR/transformers" \
    --env "HUGGINGFACE_HUB_CACHE=$HF_CACHE_DIR/hub" \
    --env "HF_DATASETS_CACHE=$HF_CACHE_DIR/datasets" \
    $HOME/pytorch_24.09.sif \
    torchrun --standalone --nproc_per_node=8 train_sae_ddp.py
