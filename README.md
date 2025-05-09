KPerfIR Artifact for OSDI'25
==============

# 1. Overview

The KPerfIR project is a performance tool infrastructure for the [Triton](https://github.com/triton-lang/triton) compiler.
Its integration into the main branch is ongoing on this [development branch](https://github.com/triton-lang/triton/tree/proton-dev).

The results for the OSDI'25 submission are derived from specific early feature branches.
To reproduce the main results, we check out these branches, some of which have been merged or refactored into the development branch.

Note that our evaluation for the FlashAttention-3 kernel requires an Nvidia H100 GPU.
We can provide a temporary evaluation account on our cluster if needed.

# 2. Installation

To evaluate the artifact, we provide a Docker image that contains the required environment and scripts.
Please run the following commands in the artifact folder to build and start the Docker container.
Since we are building the Triton compiler and baselines, the installation process may take some time (approximately 15 minutes on our machine).

```
docker build -t kperfir_artifact .
```

Then, start the Docker container:

```
docker run -it --rm --gpus all kperfir_artifact
```

# 3. Evaluation

## 3.1 Usage and Documentation
We will maintain [formal documentation](https://triton-lang.org/main/dialects/ProtonOps.html) within the official Triton documentation.
Several usage examples, including GEMM and ADD kernels, can be found [here](https://github.com/fywkevin/triton/tree/fywkevin/poc-profile/third_party/proton/tutorials/intra_kernel).
The overhead results are evaluated using NCU profiling on these scripts, which require manual effort. 
Our backend implementation has evolved with additional optimizations.
So we primarily reproduce Fig. 11 and Fig. 12, which demonstrate the usability of the performance tool and the improvements discussed in our paper.

## 3.2 Fig. 11

To profile the Triton FlashAttention kernels with KPerfIR's timing tool, run the following commands to produce the trace files shown in Fig. 11:

```
cd /workspace/tritonbench
```

For the vanilla FA3 baseline:

```
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=/workspace/kperfir_artifact/ttgir/vanilla CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a python ./run.py --op flash_attention --only triton_tutorial_flash_v2_tma_ws --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
```

For the improved FA3 implementation:

```
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=/workspace/kperfir_artifact/ttgir/improved CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a WITH_TMA=1 ENABLE_COMPPIPE=1 SWP_FOR_CONSUMER=1 PEEL_LAST_ITER=1 python ./run.py --op flash_attention --only triton_tutorial_flash_v2_tma_ws --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
```

The resulting trace file is saved as `chrome_trace.json` in the current folder.
To inspect the trace file, download it to a local machine and use the `chrome://tracing` or `edge://tracing` tool in your browser to open the JSON file.

## 3.3 Fig. 12

The FA3 kernels discussed in our manuscript demonstrate a compute pipeline improvement guided by the previous profiling results.
A comprehensive version of the FA3 kernel, including the discussed improvements, is available in this [repository](https://github.com/manman-ren/triton).

### 3.3.1 Triton-FA3
We evaluate a vanilla implementation of FA3 in Triton and an improved version using the [tritonbench repository](https://github.com/pytorch-labs/tritonbench).
These two baselines are denoted as Triton-FA3 and Triton-FA3 Improved in Fig. 12.
The implementation can be found at `/workspace/tritonbench/tritonbench/kernels/triton_fused_attention.py`.

We provide a script to reproduce the results in Fig. 12.
The script accepts two arguments: `run_type` and `benchmark`.
- `run_type`: Specifies whether to run the vanilla implementation (`vanilla`) or the improved implementation (`improved`).
- `benchmark`: Specifies whether to run the sequence length benchmark (`seq_len`) on the left side of Fig. 12 or the batch size benchmark (`batch_size`) on the right side of Fig. 12.

To reproduce the full Fig. 12:

```
cd /workspace/tritonbench

# Purple bars in Fig. 12 (left)
bash bench_fa.sh vanilla seq_len

# Yellow bars in Fig. 12 (left)
bash bench_fa.sh improved seq_len

# Purple bars in Fig. 12 (right)
bash bench_fa.sh vanilla batch_size

# Yellow bars in Fig. 12 (right)
bash bench_fa.sh improved batch_size
```

### 3.3.2 FA2 and FA3

The baselines for FA2 and FA3 (denoted as FA2 and FA3) can be evaluated using the following script.
Note that this script is the [official evaluation script](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/benchmark_attn.py) from the FA repository:

```
cd /workspace/kperfir_artifact
python benchmark_attn.py seq_len
python benchmark_attn.py batch_size
```
