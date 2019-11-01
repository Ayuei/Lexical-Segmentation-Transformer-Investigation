#!/bin/sh

set -e

sbatch runner.sh pubmed_aa
sbatch runner.sh pubmed_ab
sbatch runner.sh pubmed_ac
sbatch runner.sh pubmed_ad
