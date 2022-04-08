#!/bin/bash

CURRENT_TASK_VALUES=0
for (( RUN_IDX = 1; RUN_IDX < 200; ++RUN_IDX )); do
  CURRENT_TASK_VALUES=$CURRENT_TASK_VALUES,$(( 1 * RUN_IDX )); done

EXCLUDE_NODES="matrix-0-24,matrix-2-21,matrix-1-4,matrix-2-7,matrix-1-18,\
matrix-2-15,matrix-2-9,matrix-2-5,matrix-2-13,matrix-2-11,matrix-2-3,matrix-1-12"

SINGULARITY_COMMAND="python train_SSR_main.py \
--config_file ./SSR/configs/thor_config.yaml \
--dataset_type thor \
--save_dir /home/btrabucc/nerf-unshuffle-CURRENT_TASK \
--dataset_dir /home/btrabucc/nerf-thor/thor-unshuffle-val-CURRENT_TASK.pkl"

spork remote --num-cpus 8 --num-gpus 1 --memory 16 --num-hours 72 \
  --sweep-params CURRENT_TASK --sweep-values $CURRENT_TASK_VALUES \
  --exclude-nodes "$EXCLUDE_NODES" "$SINGULARITY_COMMAND"
