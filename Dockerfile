FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
WORKDIR /workspaces/ocr-c++

ENV build_from_docker_file 1

# install generic tools
RUN apt update && apt -y dist-upgrade && \
DEBIAN_FRONTEND="noninteractive" apt install -y wget build-essential cmake \
gdb git git-lfs libssl-dev pkg-config unzip libopencv-dev python3-opencv

# download libtorch
RUN mkdir -p /workspaces/ocr-c++/thirdparty
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.1%2Bcu118.zip -O /workspaces/ocr-c++/thirdparty/libtorch.zip
RUN unzip /workspaces/ocr-c++/thirdparty/libtorch.zip -d /workspaces/ocr-c++/thirdparty/ && rm /workspaces/ocr-c++/thirdparty/libtorch.zip

# keep container running after start
ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]