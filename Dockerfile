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
RUN pip3 install tornado
RUN pip3 install subword-nmt

# cd parlai/chat_service/services/browser_chat
# python3 run.py --config-path ../../tasks/chatbot/config.yml --port 10001
# <SEPARATE TERMINAL> python3 client.py --port 10001

