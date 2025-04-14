#!/bin/bash                                                                                                               

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=simclr_training
#SBATCH --mem=50G
#SBATCH --partition=type_2
#SBATCH --array=0-26
#SBATCH --output=simclr_%A_%a.out
#SBATCH --error=simclr_%A_%a.err

# Load required modules
module load userspace/tr17.10
module load biology
module load lapack/3.7.1 
module load jags/4.3.0
module load gcc/7.2.0 
module load proj.4/4.9.3 
module load geos/3.6.2
module load userspace
module load python

# Define grid
TEMPS=(0.2 0.5 1.0)
BATCHES=(32 64 128)
EPOCHS=(10 20 50)

i=$SLURM_ARRAY_TASK_ID

# Compute grid values
temp_idx=$((i % 3))
batch_idx=$(((i / 3) % 3))
epoch_idx=$((i / 9))

TEMP=${TEMPS[$temp_idx]}
BATCH=${BATCHES[$batch_idx]}
EPOCH=${EPOCHS[$epoch_idx]}

MODEL_NAME="model_temp${TEMP}_batch${BATCH}_epoch${EPOCH}.pth"
METRIC_LOG="logs/simclr_metrics.csv"

# Ensure output directories exist
mkdir -p models logs

echo "Running with TEMP=$TEMP, BATCH=$BATCH, EPOCH=$EPOCH"

python train_simclr.py \
    --data_path "/path/to/your/dataset" \
    --output_model_path "models/$MODEL_NAME" \
    --temperature $TEMP \
    --batch_size $BATCH \
    --epochs $EPOCH \
    --metrics_csv "$METRIC_LOG"
