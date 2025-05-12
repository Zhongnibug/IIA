#!/bin/bash

# Define Python and args
declare -a python_files=(
    "main.py -w msvd,msvd_clip,msvd_struct_IIA,diff_seed_avg,train -b"
    "main.py -w msvd,msvd_clip,msvd_seman_IIA,diff_seed_avg,train -b"
    "main.py -w msrvtt,msrvtt_clip,msrvtt_struct_IIA,diff_seed_avg,train -b"
    "main.py -w msrvtt,msrvtt_clip,msrvtt_seman_IIA,diff_seed_avg,train -b"
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