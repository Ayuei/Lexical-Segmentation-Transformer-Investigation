#!/bin/sh

#SBATCH -t 1100
#SBATCH -c 19

source /scratch1/ngu143/venv/bin/activate

python3 process_text.py "${1}"
