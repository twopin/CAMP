# Docker inheritance
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
MAINTAINER "Dominick Lemas"
RUN apt-get update 
RUN	apt-get install -y asciinema  
RUN	apt-get install -y	python2.7
RUN apt-get install unzip
RUN apt-get install -y python3-pip
RUN pip install rdkit-pypi
RUN pip install keras
RUN pip install tensorflow
COPY . /home/camp
WORKDIR /home/camp

