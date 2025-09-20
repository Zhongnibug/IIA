#!/bin/bash

# Define Python and args
declare -a python_files=(
    "main.py -w msvd,msvd_clip,Baseline,diff_seed_avg,train -b"
    "main.py -w msrvtt,msrvtt_clip,Baseline,diff_seed_avg,train -b"
)

# Foreach Python
for item in "${python_files[@]}"
do
  echo "Doing: python $item"
  python $item
  if [ $? -ne 0 ]; then
      echo "Failed to do $item!"
      exit 1
  fi
done

echo "Batch python End!"