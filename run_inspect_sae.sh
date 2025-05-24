#!/bin/bash
###############################################################################
#  File:     run_inspect_sae.sbatch
#  Purpose:  Launch inspect_sae.py as a background job with Slurm / sbatch
#            on the AAU AI-LAB cluster.
###############################################################################

############################
# 1) Slurm resource request
############################
#SBATCH --job-name=sae_inspect                 # appears in squeue
#SBATCH --gres=gpu:5                           # 8× GPUs – change if you need fewer
#SBATCH --cpus-per-task=16                     # CPU helpers for dataloading
#SBATCH --mem=64G                              # system RAM
#SBATCH --time=12:00:00                        # HH:MM:SS  (wall-clock limit)
#SBATCH --output=logs/%x-%j.out                # stdout  ( %x = job-name, %j = job-ID )
#SBATCH --error=logs/%x-%j.err                 # stderr
## optional e-mail notifications
#SBATCH --mail-user=dmarus21@student.aau.dk
#SBATCH --mail-type=END,FAIL

############################
# 2) Environment / modules
############################
# Slurm launches the job in your $HOME.  Jump to the project folder:
WORKDIR=$HOME/AIAS2/AIAS2
cd "$WORKDIR"  || { echo "Workdir not found"; exit 1; }

# (If the cluster uses Environment Modules, load them here – otherwise skip)
# module load cuda/12.2  gcc/11.3  singularity/4.0   # ← example

############################
# 3) Optional caching tweaks
############################

# Make sure SCRATCH is defined properly - use home directory if not
if [ -z "$SCRATCH" ]; then
    export SCRATCH=$HOME/scratch
fi

# Create cache directories with absolute paths
HF_CACHE_DIR="$HOME/scratch/huggingface"
mkdir -p "$HF_CACHE_DIR/hub"
mkdir -p "$HF_CACHE_DIR/transformers"
mkdir -p "$HF_CACHE_DIR/datasets"

# Debug information
echo "Setting up Hugging Face cache at: $HF_CACHE_DIR"

############################
# 4) Launch the actual job
############################
CKPT=$WORKDIR/checkpoints/2048_sae_layer12.pt
IMG_DIR=$WORKDIR/openimages_subset/images_val
OUT_DIR=$WORKDIR/sae_inspect
TOPK=40

# Pass environment variables to Singularity container with --env
singularity exec --nv \
    --bind "$HF_CACHE_DIR:$HF_CACHE_DIR" \
    --env "HF_HOME=$HF_CACHE_DIR" \
    --env "TRANSFORMERS_CACHE=$HF_CACHE_DIR/transformers" \
    --env "HUGGINGFACE_HUB_CACHE=$HF_CACHE_DIR/hub" \
    --env "HF_DATASETS_CACHE=$HF_CACHE_DIR/datasets" \
    $HOME/pytorch_24.09.sif \
    python inspect_sae.py \
        --ckpt      "$CKPT" \
        --image_dir "$IMG_DIR" \
        --layer     11 \
        --hidden 2048 \
        --top_k     "$TOPK" \
        --out_dir   "$OUT_DIR" \
        --auto_label