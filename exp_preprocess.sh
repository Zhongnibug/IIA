#!/bin/bash

# Define Python and args
declare -a python_files=(
    "main.py -w msvd,bleu_score,msvd_struct_pair,empty -b"
    "main.py -w msvd,cider_score,msvd_seman_pair,empty -b"
    "main.py -w msrvtt,bleu_score,msrvtt_struct_pair,empty -b"
    "main.py -w msrvtt,cider_score,msrvtt_seman_pair,empty -b"
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