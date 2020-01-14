# Manual

- sources.list用https
- [error](https://blog.csdn.net/lqsnjust/article/details/81129129)
```
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

- jupyter notebook --generate-config
   - Writing default config to: /root/.jupyter/jupyter_notebook_config.py
     - c.NotebookApp.ip = '*'
     - c.NotebookApp.port = 8888
     - c.NotebookApp.notebook_dir = '/home/kindlytree'
   - jupyter notebook password(input i0****st)
   - jupyter notebook --allow-root