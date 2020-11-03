#!/bin/csh
#SBATCH -J reqm_AuCO_01007
#SBATCH -o reqm_min.out
#SBATCH -e reqm_min.err
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 05:00:00
##SBATCH --mail-user=naserifar.saber@gmail.com
##SBATCH --mail-type=END,FAIL   # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH -A projectnumber
pwd; hostname; date
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"= $SLURM_JOB_NODELIST
/home/naseri/miniconda3/bin/python reqm.py > LOG_corrected
date
