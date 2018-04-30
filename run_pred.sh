#!/bin/bash
#SBATCH -J deepGO
#SBATCH -p gpu_q
##SBATCH -p pegasus_q
#SBATCH --time=18:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
##SBATCH --nodes=1 --ntasks-per-node=10 
#SBATCH --mem=50G
#module load CUDA/8.0.44
#module load cudnn/7.0
#module load Python/3.6.4-foss-2017a
module load TensorFlow
source /home/sathap1/.start_pegasus.sh
source /home/sathap1/pegasus_local/venvs/deepgo/bin/activate
#pip install tensorflow-gpu==1.2.0 # --upgrade
#pip install obonet --user
export LD_LIBRARY_PATH=/home/sathap1/pegasus_local/cudnn51/lib64/:$LD_LIBRARY_PATH
export CUDNN_DIR=/home/sathap1/pegasus_local/cudnn51/
export CUDNN_INC=${CUDNN_DIR}/include
export CUDNN_LIB=$CUDNN_DIR/lib64
export INCLUDE=$CUDNN_INC:$INCLUDE

#python test.py

#python ./fungcat_seq_newKeras.py --function cc --train  
#python ./fungcat_seq_predict.py --function cc --train  
python ./deepGO.py --function bp --data ./resources --outdir ./results 
#python ./fungcat_get_functions.py

deactivate
