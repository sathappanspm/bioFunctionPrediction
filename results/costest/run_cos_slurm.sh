#!/bin/bash
#SBATCH -J RNNSEQ
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

## Modules for discovery
#module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
#source ~/.start_discovery.sh



LOCAL="workspace"
#### Code for running python
RESULTDIR="${HOME}/${LOCAL}/bioFunctionPrediction/results/costest"
SCRIPT_ROOT="${HOME}/${LOCAL}/bioFunctionPrediction/src/"
cd $SCRIPT_ROOT
DATA="${HOME}/${LOCAL}/bioFunctionPrediction/data/data_BP/"
OUTDIR="${RESULTDIR}/model_noLabelTraining_$( date -I)"
mkdir -p $OUTDIR

BATCHSIZE=64
python ./cosEmbedding.py --function '' --resources ${SCRIPT_ROOT}/../resources --inputfile ${DATA} --batchsize $BATCHSIZE --featuretype ngrams --maxseqlen 2002  --outputdir $OUTDIR --trainsize $(( 18815 / $BATCHSIZE )) --testsize $(( 5846 / $BATCHSIZE )) --validationsize $(( 4704 / $BATCHSIZE )) --num_epochs 20


