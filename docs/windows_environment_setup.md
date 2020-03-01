# Windows environment setup
- anaconda instatllation
    - [download](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
    - conda安装pytorch
        - 安装后进入Anaconda Prompt
        - conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        - conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
        - conda config --set show_channel_urls yes
        - conda install pytorch torchvision -c pytorch
