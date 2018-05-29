#!/bin/bash
# Resources can be requested by specifying the number of nodes, cores, memory, GPUs, etc
# Examples:
#   Request 2 nodes with 24 cores each
#   #PBS -l nodes=1:ppn=24
#   Request 4 cores (on any number of nodes)
#   #PBS -l procs=4
#   Request 12 cores with 20gb memory per core
#   #PBS -l procs=12,pmem=20gb
#   Request 2 nodes with 24 cores each and 20gb memory per core (will give two 512gb nodes)
#   #PBS -l nodes=2:ppn=24,pmem=20gb
#   Request 2 nodes with 24 cores per node and 1 gpu per node
#   #PBS -l nodes=2:ppn=24:gpus=1
#   Request 2 cores with 1 gpu each
#   #PBS -l procs=2,gpus=1
#   #PBS -l nodes=1:ppn=24

#PBS -l nodes=1:gpus=1

#PBS -l walltime=03:00:00

#### Queue ####
# Queue name. NewRiver has seven queues:
#   normal_q        for production jobs on all Haswell nodes (nr003-nr126)
#   largemem_q      for jobs on the two 3TB, 60-core Ivy Bridge servers (nr001-nr002)
#   dev_q           for development/debugging jobs on Haswell nodes. These jobs must be short but can be large.
#   vis_q           for visualization jobs on K80 GPU nodes (nr019-nr027). These jobs must be both short and small.
#   open_q          for jobs not requiring an allocation. These jobs must be both short and small.
#   p100_normal_q   for production jobs on P100 GPU nodes
#   p100_dev_q      for development/debugging jobs on P100 GPU nodes. These jobs must be short but can be large.
# For more on queues as policies, see http://www.arc.vt.edu/newriver#policy

#PBS -q p100_normal_q

#### Account ####
# This determines which allocation this job's CPU hours are billed to.
# Replace "youraccount" below with the name of your allocation account.
# If you are a student, you will need to get this from your advisor.
# For more on allocations, go here: http://www.arc.vt.edu/allocations

#PBS -A fungcat1

# Access group. Do not change this line.

#PBS -W group_list=newriver

# Uncomment and add your email address to get an email when your job starts, completes, or aborts

#PBS -M sathap1@vt.edu

# #PBS -m bea

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR
module purge
module load Anaconda
#module load Anaconda/2.3.0
module load cuda/8.0.44
module load cudnn/6.0
source activate /home/sathap1/newriver_local/venvs/tfgpu
OUTDIR="/home/sathap1/workspace/bioFunctionPrediction/results/deepGO"
mkdir -p $OUTDIR
#export INCLUDE=$CUDNN_INC:$INCLUDE
python ./deepGO.py --function bp --data ./resources --outputdir $OUTDIR --trainsize 2000 --testsize 2000 --validationsize 10
deactivate
