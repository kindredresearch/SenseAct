FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         nano \
         python3-dev \ 
         python3-pip \
         python3-setuptools \
         libopenmpi-dev \
         openmpi-bin \
         openssh-client \
         openssh-server \
         net-tools \
         iputils-ping \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN git clone https://github.com/dti-research/SenseAct && cd SenseAct && pip3 install -e .

# Exit 0 hack for forced MuJoCo installation which is obviously missing.
RUN pip3 install baselines==0.1.5; exit 0
