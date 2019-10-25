FROM ubuntu:latest

# set Arm version to use ( 'armv7a', 'arm64-v8a', 'arm64-v8.2-a', 'arm64-v8.2-a-sve' )
ARG ARCH=armv7a

# base dir setting
ARG BASEDIR="/root/armnn-onnx"

# uncomment lines below to compile for opencl
#ARG OPENCL_SCONS="opencl=1 embed_kernels=1"
#ARG OPENCL_ARMNN="-DARMCOMPUTECL=1"

# install dependencies via apt
RUN apt update && \
    apt install -y \
    # packages to install here
    autoconf \
    automake \
    cmake \
    g++ \
    gcc \
    git \
    libtool \
    scons \
    wget \
  && rm -rf /var/lib/apt/lists/*

# prepare sources
#ENV BASEDIR=${BASEDIR}
RUN [ "/bin/bash", "-c", "mkdir -p $BASEDIR" ]
WORKDIR ${BASEDIR}
# Git part
RUN [ "/usr/bin/git", "clone", "https://github.com/Arm-software/ComputeLibrary.git" ]
RUN [ "/usr/bin/git", "clone", "https://github.com/Arm-software/armnn" ]
RUN [ "/usr/bin/git", "clone", "https://github.com/google/protobuf.git" ]
# tar part
RUN [ "/bin/bash", "-c", "set -o pipefail && wget -qO- https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2 | tar xjf -" ]

# building ComputeLibrary
WORKDIR ${BASEDIR}/ComputeLibrary
RUN [ "/bin/bash", "-c", "scons arch=$ARCH extra_cxx_flags=\"-fPIC\" benchmark_tests=0 validation_tests=0 neon=1 $OPENCL_SCONS -j $(nproc)" ]

# building Boost Library
WORKDIR ${BASEDIR}/boost_1_64_0
RUN [ "/bin/bash", "-c", "./bootstrap.sh && ./b2 --build-dir=$BASEDIR/boost_1_64_0/build toolset=gcc link=static cxxflags=-fPIC --with-filesystem --with-test --with-log --with-program_options install --prefix=$BASEDIR/boost" ]

# building Protobuf
WORKDIR ${BASEDIR}/protobuf
RUN [ "/bin/bash", "-c", "git submodule update --init --recursive && ./autogen.sh && ./configure --prefix=$BASEDIR/protobuf-host && make -j $(nproc) && make install" ]

# Generate ONNX protobuf source files
WORKDIR ${BASEDIR}
RUN [ "/bin/bash", "-c", "export ONNX_ML=1; git clone --recursive https://github.com/onnx/onnx.git; unset ONNX_ML" ]
WORKDIR ${BASEDIR}/onnx
RUN [ "/bin/bash", "-c", "export LD_LIBRARY_PATH=$BASEDIR/protobuf-host/lib:$LD_LIBRARY_PATH; \
                          $BASEDIR/protobuf-host/bin/protoc onnx/onnx.proto \
                                                            --proto_path=. \
                                                            --proto_path=$BASEDIR/protobuf-host/include --cpp_out $BASEDIR/onnx" ]

# Build ARM NN
WORKDIR ${BASEDIR}/armnn
RUN [ "/bin/bash", "-c", "mkdir build" ]
WORKDIR ${BASEDIR}/armnn/build
# copy into current mobilenet modified file
COPY OnnxMobileNet-Armnn.cpp $BASEDIR/armnn/tests/OnnxMobileNet-Armnn/OnnxMobileNet-Armnn.cpp
# patch deprecated code
RUN sed -i 's/codedStream\.SetTotalBytesLimit/\/\/codedStream\.SetTotalBytesLimit/g' $BASEDIR/armnn/src/armnnOnnxParser/OnnxParser.cpp
RUN [ "/bin/bash", "-c", "cmake .. -DBOOST_ROOT=$BASEDIR/boost \
                                   -DPROTOBUF_ROOT=$BASEDIR/protobuf-host \
                                   -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
                                   -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build \
                                   -DBUILD_ONNX_PARSER=1 -DONNX_GENERATED_SOURCES=$BASEDIR/onnx \
                                   -DARMCOMPUTENEON=1 -DARMNNREF=1 -DBUILD_TESTS=1 -DARMNNREF=1 $OPENCL_ARMNN && \
                          make -j $(nproc)" ]

# Unit Test will fail 
# RUN [ "/bin/bash", "-c", "./UnitTests" ]

ENTRYPOINT [ "/bin/bash" ]
