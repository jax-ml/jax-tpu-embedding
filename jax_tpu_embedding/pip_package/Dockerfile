ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="JAX TPU Embedding <jax-tpu-embedding-dev@google.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
ARG wheel_folder
ENV WHEEL_FOLDER $wheel_folder
ENV PYTHON_VERSION="3"
ENV PYTHON_MINOR_VERSION="8"

# Pick up some TF dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends software-properties-common
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        aria2 \
        build-essential \
        curl \
        dirmngr \
        emacs \
        git \
        gpg-agent \
        less \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        lsof \
        pkg-config \
        rename \
        rsync \
        python-dev \
        python3-distutils \
        sox \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip python3.8-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.8 -m pip install pip --upgrade && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0

# Install pip package
RUN pip3 install pip-tools
RUN pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install tensorflow~=2.9.2

ARG bazel_version=5.1.1
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

COPY . /jax_tpu_embedding
RUN mkdir -p $WHEEL_FOLDER && cd jax_tpu_embedding && git rev-parse HEAD > $WHEEL_FOLDER/jax_tpu_embedding_commit.txt

RUN pip3 install pip six numpy wheel setuptools mock 'future>=0.17.1'

RUN mv jax_tpu_embedding/jax_tpu_embedding/pip_package /jax_tpu_embedding/
RUN cd /jax_tpu_embedding && bash pip_package/build.sh

WORKDIR /

CMD ["/bin/bash"]
