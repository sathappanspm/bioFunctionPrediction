#!/bin/bash
#SBATCH -J multiCNN
#SBATCH -p pegasus_q
##SBATCH -p pegasus_q
#SBATCH --time=18:00:00
##SBATCH --nodes=2  --gres=gpu:1
#SBATCH --nodes=20
#SBATCH --mem=30G
module load Python/3.6.4-foss-2017a

#module load cuda90
#module load cudnn/7.0
#module load TensorFlow
source /home/sathap1/.start_pegasus.sh
source /home/sathap1/pegasus_local/venvs/tfcpu/bin/activate
pip install tensorflow 
pip install obonet
OUTDIR="results_multiCNN"
mkdir -p $OUTDIR
#export LD_LIBRARY_PATH=/home/sathap1/pegasus_local/cudnn51/lib64/:$LD_LIBRARY_PATH
#export CUDNN_DIR=/home/sathap1/pegasus_local/cudnn51/
#export CUDNN_INC=${CUDNN_DIR}/include
#export CUDNN_LIB=$CUDNN_DIR/lib64
#export INCLUDE=$CUDNN_INC:$INCLUDE
python ./multicharcnn_run.py --function bp --data ./resources --outputdir $OUTDIR --trainsize 6000 --testsize 2000 --validationsize 100
deactivate
