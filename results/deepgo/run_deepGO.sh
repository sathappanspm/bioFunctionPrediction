#!/bin/bash
#SBATCH -J PFP
##SBATCH -p normal_q
#SBATCH -p gpu_q
#SBATCH -n 1
#SBATCH -A fungcat
##SBATCH --nodelist hu008
#SBATCH -t 360:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1

#module load anaconda2
#module load cuda
#source activate pytorch

module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
source ~/.start_discovery.sh
RESULTDIR="/home/sathap1/workspace/bioFunctionPrediction/results/deepgo/"
SCRIPT_ROOT="/home/sathap1/workspace/bioFunctionPrediction/src/"
cd $SCRIPT_ROOT

FUNCTION="bp"
OUTDIR="${RESULTDIR}/model_3mers_${FUNCTION}"
mkdir -p $OUTDIR

BATCHSIZE=16

python ${SCRIPT_ROOT}/deepGO.py --resources ${SCRIPT_ROOT}/../resources --outputdir $OUTDIR --trainsize $(( 91987 / $BATCHSIZE )) --testsize $(( 24205 / $BATCHSIZE )) --validationsize $(( 4841 / $BATCHSIZE )) --inputfile ${SCRIPT_ROOT}/../resources/data --batchsize $BATCHSIZE --featuretype ngrams --maxseqlen 2002 --function ${FUNCTION}

source deactivate
