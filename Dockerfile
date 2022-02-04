# Docker inheritance
FROM continuumio/miniconda3:latest
MAINTAINER "Dominick Lemas"
RUN apt-get update
RUN apt-get -y install python2.7 
RUN conda install -y -c conda-forge rdkit
RUN pip install keras
RUN pip install tensorflow