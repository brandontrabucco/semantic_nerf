#!/bin/bash

SINGULARITY_COMMAND="python -u agent.py --fake-home \
--logdir /home/btrabucc/new-fine-tuned --start-task START_TASK --total-tasks 50"

START_TASK_VALUES=0
for (( RUN_IDX = 1; RUN_IDX < 80; ++RUN_IDX )); do
  START_TASK_VALUES=$START_TASK_VALUES,$(( 50 * RUN_IDX )); done

EXCLUDE_NODES="matrix-0-24,matrix-1-16,matrix-2-21,matrix-1-4,matrix-2-7,\
matrix-2-15,matrix-2-9,matrix-2-5,matrix-2-13,matrix-2-11,matrix-2-3,matrix-1-12"

spork remote --num-cpus 8 --num-gpus 1 --memory 16 --num-hours 72 \
  --sweep-params START_TASK --sweep-values $START_TASK_VALUES \
  --exclude-nodes "$EXCLUDE_NODES" "$SINGULARITY_COMMAND"
