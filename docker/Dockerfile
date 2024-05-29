# Dockerfile to run the diffusion models on GPULab
# make sure to mount a data volume to /data when spinning up the image
# and to have the renders there as well.


FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    rsync && \
    rm -rf /var/lib/apt/lists/*
# # instal the dependencies
#RUN pip install torch torchvision torchaudio
RUN pip install diffusers transformers click tqdm controlnet_aux accelerate
#RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# remove the opencv-python and install opencv-python-headless
RUN pip uninstall opencv-python -y
RUN pip3 install opencv-python-headless


# copy the code

COPY airo-blender/airo-mono/airo-typing/ /airo-mono/airo-typing
COPY airo-blender/airo-mono/airo-spatial-algebra/ /airo-mono/airo-spatial-algebra
COPY airo-blender/airo-mono/airo-dataset-tools/ /airo-mono/airo-dataset-tools

RUN pip install -e /airo-mono/airo-typing
RUN pip install -e /airo-mono/airo-spatial-algebra
RUN pip install -e /airo-mono/airo-dataset-tools

# uninstall opencv and install opencv-headless
RUN pip uninstall opencv-python -y
RUN pip install opencv-python-headless==4.7.0.68

COPY dsd/dsd /dsd/dsd
COPY dsd/setup.py /dsd/setup.py
COPY dsd/experiments /dsd/experiments
RUN pip install -e /dsd

# link the project mount
# RUN ln -s /project_antwerp/dsd/data /data

# set hugginface caching dir
# to avoid going over 10GB limit
# https://huggingface.co/docs/huggingface_hub/guides/manage-cache
ENV HF_HUB_CACHE=/data/.cache/HF
WORKDIR /dsd



