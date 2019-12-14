FROM ubuntu:16.04
# RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
# RUN apt-get clean
#RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list
#RUN sed -i 's/http:\/\/security.ubuntu.com\/ubuntu\//http:\/\/mirrors.security.aliyun.com\/ubuntu\//g' /etc/apt/sources.list
ADD sources.list /etc/apt/

RUN apt-get update

ADD requirements.txt /home/requirements.txt
RUN apt-get install python-software-properties  --assume-yes
RUN apt-get install software-properties-common --assume-yes
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install python3.6  --assume-yes
RUN apt-get update && apt-get install -y python3-pip python3-tk python3-lz4 libsasl2-dev 
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip3 install  -r /home/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN apt-get -y install openjdk-8-jdk git

WORKDIR /home/kindlytree