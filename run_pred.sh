#!/bin/bash
#SBATCH -J deepGO
#SBATCH -p gpu_q
##SBATCH -p pegasus_q
#SBATCH --time=18:00:00
#SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --mem=50G
module load Python/3.6.4-foss-2017a
module load cuda90
module load cudnn/7.0
#module load TensorFlow
source /home/sathap1/.start_pegasus.sh
source /home/sathap1/pegasus_local/venvs/tfnew/bin/activate
#export LD_LIBRARY_PATH=/home/sathap1/pegasus_local/cudnn51/lib64/:$LD_LIBRARY_PATH
#export CUDNN_DIR=/home/sathap1/pegasus_local/cudnn51/
#export CUDNN_INC=${CUDNN_DIR}/include
#export CUDNN_LIB=$CUDNN_DIR/lib64
#export INCLUDE=$CUDNN_INC:$INCLUDE

python ./deepGO.py --function bp --data ./resources --outdir ./results --trainsize 10 --testsize 10 --validationsize 2 
deactivate
