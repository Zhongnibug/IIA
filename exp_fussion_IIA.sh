#!/bin/bash

# Define Python and args
declare -a python_files=(
    "main.py -w msvd,msvd_clip,msvd_IIA,diff_seed_avg,IIA_load,test -b"
    "main.py -w msrvtt,msrvtt_clip,msrvtt_IIA,diff_seed_avg,IIA_load,test -b"
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