# Docker inheritance
FROM continuumio/miniconda3:latest
MAINTAINER "Dominick Lemas"
RUN apt-get update
RUN apt-get -y install python2.7 python-rdkit librdkit1 rdkit-data
RUN pip install keras
RUN pip install tensorflow