#!/bin/bash
#SBATCH --job-name=4_timing_1
#SBATCH --nodes=1
#SBATCH --mem=170G
#SBATCH --output=4_timing.%J.out
#SBATCH --error=4_timing.%J.err
#SBATCH -p ctessum 
#SBATCH --gres=gpu:QuadroRTX6000:1
#SBATCH --time=72:00:00

cd /projects/illinois/eng/cee/ctessum/xiaokai2/partmc-mcm/MCM-PartMC-MOSAIC-master/script/7.1_chem_mech5/2_SINDy/2_PartMC-MCM/case_6000_15d_real_emit_puremcm/2_simady/Computational_Speedup/code/gpu/code/1
source activate SIMADy
julia 1_timing.jl
