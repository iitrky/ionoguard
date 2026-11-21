#!/usr/bin/env bash
# Full pipeline used for the paper (two workers in parallel).
set -e
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/scripts"
[ -f ../results/models_S3.pkl ] || python train_models.py   # trained models are included
python run_ext.py A & python run_ext.py B; wait
python run_sweeps.py && python run_trace.py && python timing.py && python summarize.py
python summarize_ext.py
python run_ol.py A & python run_ol.py B; wait
python run_mu_ol.py A & python run_mu_ol.py B; wait
python summarize_ol.py
python make_figures.py && python make_block_diagrams.py && python make_system_diagrams.py \
  && python make_mu_diagram.py && python make_new_figures.py
