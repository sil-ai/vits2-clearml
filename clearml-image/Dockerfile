FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Install python
RUN apt-get update && apt-get install -y python3 python3-pip wget git sox libsox-dev espeak-ng

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and TorchText
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 torchtext==0.18.0
