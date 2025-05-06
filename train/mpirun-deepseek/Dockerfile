FROM nvcr.io/nvidia/pytorch:25.01-py3 as base

ENV SHELL /bin/bash

RUN rm /opt/megatron-lm -rf
RUN apt-get update
RUN apt-get install -y sudo gdb pstack bash-builtins git zsh autojump tmux curl
RUN pip install debugpy dm-tree torch_tb_profiler einops wandb
RUN pip install sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic==2.2.1
RUN pip install nvidia-pytriton py-spy yapf darker pytest-cov pytest_mock
RUN TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# envsubst used for model_params substitution
RUN apt-get install -y gettext

# yq is a lightweight and portable command-line YAML processor; install yq through snapd
RUN wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

RUN pip3 install --no-cache-dir \
      black==24.4.2 \
      isort==5.13.2 \
      flake8==7.1.0 \
      pylint==3.2.6

# Install TE w/ NVTE_WITH_USERBUFFERS=1 so that `--tp-comm-overlap` can be enabled
RUN NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.13

# Install onelogger
RUN pip install --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple one-logger

RUN pip install setuptools==69.5.1
