# Manual

## Environment requirements and setup
 The requirements of environment of this repo are in two aspects(please refer [INSTALL](https://github.com/kindlytree/sys_tools/blob/master/README.md) to the installation of the environment):
 - Ｈardware: Nvidia GPU device
 - Software: 
    - System: Ubuntu
    - Applications: docker(nvidia-docker2) 

- Docker environment setup
  - cd docker/dl/pytorch
  - execute docker-compose build
  - execute docker-compose up -d
  - execute 'sudo ./start.sh' to set jupyter notebook(input password, such as 123)

- Start Jupyter notebook
  - docker exec -it kindlytree_ai bash
  - execute "jupyter notebook --allow-root"

- Jupyter notebook Usage
  - open the web browser
  - input in address bar : localhost:1234/tree

## Samples introduction


```
legacy:
- sources.list用https
- [error](https://blog.csdn.net/lqsnjust/article/details/81129129)

W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64  Release: The following signatures were invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>
原因是相应的KEY有问题
解决办法：更新所有KEY
sudo apt-get clean
cd /var/lib/apt
sudo mv lists lists.old
sudo mkdir -p lists/partial
sudo apt-get clean
sudo apt-get update
```