FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

ADD sources.list /etc/apt/
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        vim \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

#FROM caffe-base-test2

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
# ENV CLONE_TAG=1.0
COPY . . 
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    mkdir -p /root/.config/pip && echo "[global]" >> /root/.config/pip/pip.conf && echo "index-url=https://mirrors.aliyun.com/pypi/simple/" >> /root/.config/pip/pip.conf && \
    pip install --upgrade pip && \
    pip install easydict && \
    cd lib/SSH/caffe-ssh && \
    cd python && for req in $(cat requirements.txt) pydot; do python -m pip install $req; done && cd .. && \
    make -j"$(nproc)" pycaffe

RUN for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    cd lib/SSH/lib && make && cd .. 
#    export http_proxy='http://192.168.0.119:3128' && \
#    bash scripts/download_ssh_model.sh && \
#    unset http_proxy

ENV PYCAFFE_ROOT $CAFFE_ROOT/lib/SSH/caffe-ssh/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/lib/SSH/caffe-ssh/python/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/lib/SSH/caffe-ssh/python/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

#WORKDIR /workspace
