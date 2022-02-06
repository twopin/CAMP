# Docker inheritance
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
MAINTAINER "Dominick Lemas"
RUN apt-get update 
RUN	apt-get install -y asciinema  
RUN	apt-get install -y	python2.7 
RUN apt-get install unzip
RUN conda install -y -c conda-forge rdkit
RUN pip install keras
RUN pip install tensorflow
RUN pip3 install asciinema
COPY . /home/camp
WORKDIR /home/camp

