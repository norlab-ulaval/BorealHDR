FROM ubuntu:jammy

SHELL ["/bin/bash", "-c"]

ARG USERNAME=user
ARG PASSWORD=docker
ARG MY_UID=1000

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
        build-essential \
        clang \
        cmake \
        ffmpeg \
        fuse \
        g++ \
        gcc \
        gdb \
        git \
        iputils-ping \
        libboost-all-dev \ 
        libeigen3-dev \
        libgl1-mesa-glx \
        libomp-dev \
        micro \
        python3 \
        python3-dev \
        python3-pip \
        rsync \
        screen \
        software-properties-common \
        ssh \
        sudo \
        tar \
        wget
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# add user
RUN useradd -u ${MY_UID} -m ${USERNAME} -s /bin/bash  && yes ${PASSWORD} | passwd ${USERNAME}
RUN usermod -aG sudo ${USERNAME}

# USER ${USERNAME}

# Give ssh key github
RUN mkdir /home/user/.ssh
RUN ssh-keygen -q -t rsa -N '' -f /home/user/.ssh/id_rsa

COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

USER $USERNAME
ENV HOME=/home/user

COPY ./docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]

WORKDIR /home/user/code/
