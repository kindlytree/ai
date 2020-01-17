# docker
- docker pull pytorch/pytorch
Dockerfile is supplied to build images with cuda support and cudnn v7. You can pass -e PYTHON_VERSION=x.y flag to specify which Python version is to be used by Miniconda, or leave it unset to use the default. Build from pytorch repo directory as docker needs to copy git repo into docker filesystem while building the image.

- git clone sudo git clone https://github.com/pytorch/pytorch.git
- docker build -t pytorch -f docker/pytorch/Dockerfile .  # [optional] --build-arg WITH_TORCHVISION=0
You can also pull a pre-built docker image from Docker Hub and run with nvidia-docker, but this is not currently maintained and will pull PyTorch 0.2.

nvidia-docker run --rm -ti --ipc=host pytorch/pytorch:latest