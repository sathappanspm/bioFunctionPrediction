#!/bin/bash
#SBATCH -J deepRNN
#SBATCH -p normal_q
#SBATCH -N 1
##SBATCH --nodelist hu003
#SBATCH -t 720:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:pascal:1

module load anaconda2
module load cuda
source activate pytorch

SCRIPT_ROOT="/home/sathap1/workspace/bioFunctionPrediction/src/"
OUTDIR="/home/sathap1/workspace/bioFunctionPrediction/results/locationPrediction/models"

mkdir -p $OUTDIR

cd $SCRIPT_ROOT
python ./locationPredictor.py --function '' --resources ${SCRIPT_ROOT}/../resources --outputdir $OUTDIR --trainsize 6000 --testsize 2000 --validationsize 100 --maxnumfuncs 10 --inputfile ${SCRIPT_ROOT}/../AllSeqsWithGO_expanded.tar

source deactivate
