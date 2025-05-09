#!/bin/bash

# Check if the user provided two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <run_type> <benchmark>"
    echo "run_type: 'vanilla' for the first run, 'improved' for the second run"
    echo "benchmark: 'seq_len' to iterate seq_len from 7 to 14, 'batch_size' to iterate batch_size from 1 to 128 (multiplied by 2)"
    exit 1
fi

run_type=$1
benchmark=$2

if [ "$run_type" != "vanilla" ] && [ "$run_type" != "improved" ]; then
    echo "Invalid run_type: $run_type"
    echo "Please specify 'vanilla' for the first run or 'improved' for the second run."
    exit 1
fi

if [ "$benchmark" == "seq_len" ]; then
    for seq_len in {7..14}; do
        echo "Running with seq_len=$seq_len (Benchmark: seq_len, Run Type: $run_type)"
        if [ "$run_type" == "vanilla" ]; then
            CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a  \
            python ./run.py --op flash_attention --only triton_tutorial_flash_v2_tma_ws --num-inputs 1 --seq-len $seq_len --metrics tflops --batch 16 --n-heads 16 --d-head 128
        elif [ "$run_type" == "improved" ]; then
            CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a WITH_TMA=1 ENABLE_COMPPIPE=1 SWP_FOR_CONSUMER=1 PEEL_LAST_ITER=1\
            python ./run.py --op flash_attention --only triton_tutorial_flash_v2_tma_ws --num-inputs 1 --seq-len $seq_len --metrics tflops --batch 16 --n-heads 16 --d-head 128
        fi
    done
elif [ "$benchmark" == "batch_size" ]; then
    batch_size=1
    while [ $batch_size -le 128 ]; do
        echo "Running with batch_size=$batch_size (Benchmark: batch_size, Run Type: $run_type)"
        if [ "$run_type" == "vanilla" ]; then
            CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a  \
            python ./run.py --op flash_attention --only triton_tutorial_flash_v2_tma_ws --num-inputs 1 --seq-len 12 --metrics tflops --batch $batch_size --n-heads 16 --d-head 128
        elif [ "$run_type" == "improved" ]; then
            CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a WITH_TMA=1 ENABLE_COMPPIPE=1 SWP_FOR_CONSUMER=1 PEEL_LAST_ITER=1\
            python ./run.py --op flash_attention --only triton_tutorial_flash_v2_tma_ws --num-inputs 1 --seq-len 12 --metrics tflops --batch $batch_size --n-heads 16 --d-head 128
        fi
        batch_size=$((batch_size * 2))
    done
else
    echo "Invalid benchmark: $benchmark"
    echo "Please specify 'seq_len' or 'batch_size'."
    exit 1
fi