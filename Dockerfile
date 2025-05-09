FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN pip install --no-cache-dir ninja cmake wheel pybind11 tabulate

RUN apt-get update && \
    apt-get install -y cmake build-essential git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install flashattention
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout f86e3dd && \
    pip install flash-attn --no-build-isolation && \
    cd hopper && \
    python setup.py install && \
    cd ..

# Install triton
RUN git clone https://github.com/ChandlerGuan/triton.git && \
    cd triton && \
    pip install -e python && \
    cd ..

# Install tritonbench
RUN git clone https://github.com/manman-ren/tritonbench.git && \
    cd tritonbench && \
    git checkout fa-ws-tunable && \
    cd ..

# RUN curl -L https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb     -o /tmp/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
#     DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y /tmp/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
#     rm /tmp/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb

COPY ./changes.patch /workspace/tritonbench/changes.patch
RUN cd tritonbench && \
    patch -p1 < ./changes.patch && \
    rm changes.patch && \
    cd ..

ADD . /workspace/kperfir_artifact/

RUN cp /workspace/kperfir_artifact/bench_fa.sh /workspace/tritonbench/

CMD ["bash"]