#!/bin/bash
#SBATCH -J deepAE
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH --nodelist hu010
#SBATCH -t 560:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:pascal:1

module load anaconda2
module load cuda
source activate pytorch

RESULTDIR="/home/sathap1/workspace/bioFunctionPrediction/results/rnnAE/"
SCRIPT_ROOT="/home/sathap1/workspace/bioFunctionPrediction/src/"
cd $SCRIPT_ROOT

OUTDIR="${RESULTDIR}/models"
mkdir -p $OUTDIR

BATCHSIZE=4

python ${SCRIPT_ROOT}/rnn_AE.py --resources ${SCRIPT_ROOT}/../resources --outputdir $OUTDIR --trainsize $(( 512000 / $BATCHSIZE )) --testsize $(( 256000 / $BATCHSIZE )) --validationsize 100 --inputfile ${SCRIPT_ROOT}/../AllSeqsWithGO_expanded.tar --batchsize $BATCHSIZE --featuretype ngrams

source deactivate
