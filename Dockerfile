FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update

#RUN apt-get install build-essential -y
RUN apt-get install cmake -y
RUN apt-get install make -y
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install gcc

#RUN wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
#RUN sh cuda_10.2.89_440.33.01_linux.run

RUN pip3 install torch torchvision

WORKDIR /service

RUN git clone https://github.com/facebookresearch/ParlAI.git /service/ParlAI
RUN cd /service/ParlAI; pip3 install ./requirements.txt; echo "" > README.md; python3 setup.py develop

# python3 parlai/scripts/custom_self_chat.py -t custom_skill_talk -mf zoo:blender/blender_90M/model -m transformer/custom_generator --beam-size 100
