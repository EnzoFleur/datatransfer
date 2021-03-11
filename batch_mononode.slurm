#!/bin/sh
#SBATCH --job-name=mnist_tensorflow_mononode
#SBATCH --output=mnist_tensorflow_mononode.out
#SBATCH --error=mnist_tensorflow_mononode.err
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
##SBATCH --qos=qos_gpu-dev
#SBATCH --time=00:10:00

# cleans out modules loaded in interactive and inherited by default
module purge

# loading modules
module load tensorflow-gpu/py3/2.3.1

# echo of launched commands
set -x

# code execution
srun python -u mnist-distributed.py --epochs 8 --batch-size 128
