#!/bin/bash
#SBATCH -J PFP
#SBATCH -p normal_q

## Comment this next line in huckleberry
##SBATCH -p gpu_q

#SBATCH -n 1

## Next line needed only in VBI servers
##SBATCH -A fungcat

## Check sinfo before setting this
#SBATCH --nodelist hu006

#SBATCH -t 360:00
#SBATCH --mem=30G

## Uncomment for huckleberry
#SBATCH --gres=gpu:pascal:1

## comment if not huckleberry
##SBATCH --gres=gpu:1


## ---  Modules for huckleberry, uncomment accordingly --- ##
module load anaconda2
module load cuda
module load nccl

## User specific anaconda virtual environment
#source activate venv
source activate pytorch

FUNCTION="mf"
OUTDIR="/home/sathap1/workspace/bioFunctionPrediction/results/kerasDeepGO/model_3mers_${FUNCTION}_$( date -I)"



SCRIPT_ROOT="/home/sathap1/workspace/bioFunctionPrediction/src/"
DATA="${HOME}/${LOCAL}/bioFunctionPrediction/data/data_MF/"
cd $SCRIPT_ROOT

BATCHSIZE=128

python ${SCRIPT_ROOT}/keras_deepgorun.py --function $FUNCTION --resources ${SCRIPT_ROOT}/../resources --outputdir $OUTDIR --trainsize $(( 18815 / $BATCHSIZE )) --testsize $(( 5846 / $BATCHSIZE )) --validationsize $(( 4704 / $BATCHSIZE )) --inputfile ${DATA} --batchsize $BATCHSIZE --featuretype ngrams --maxseqlen 2002

source deactivate
