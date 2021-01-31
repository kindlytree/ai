FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bk
ADD sources.list /etc/apt/

RUN apt-get clean && cd /var/lib/apt && mv lists lists.old && mkdir -p lists/partial
RUN apt-get clean
RUN apt-get update
 
RUN apt-get install nano

RUN conda install ipython
RUN conda install jupyter
RUN conda install matplotlib
RUN conda install scikit-learn
RUN pip install tensorwatch pandas -i https://pypi.douban.com/simple
# RUN chmod -R a+w .