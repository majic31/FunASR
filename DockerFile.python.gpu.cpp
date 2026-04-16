# ---------------------------------------------------------------
# ---------- 下面的必须是基础，非必须则可以删掉，跟原镜像保持一致
# ---------- 运行：docker build -f DockerFile.python.gpu.cpp  -t funasr_maj:gpu_cpp_1.2 .
# ---------------------------------------------------------------

# ubuntu基础镜像
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# 设置阿里云 apt 源 (可选)
# RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 2. 设置时区和系统依赖，防止 tzdata 卡住编译
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装系统级依赖
RUN apt-get update && \
    apt-get install -y git vim unzip iputils-ping libopenblas-dev libssl-dev cron ffmpeg tzdata \
    python3.8 python3.8-dev python3-pip curl wget cmake && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python -m pip install --upgrade pip

# 4. 先把下好的离线 wheel 包拷过去
WORKDIR /workspace
COPY torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl /workspace/
COPY torchaudio-0.12.1+cu113-cp38-cp38-linux_x86_64.whl /workspace/

# 5. 集中安装所有的 Python 依赖 (系统 pip 目录为 /usr/local/lib/python3.8/dist-packages)
# 必须先锁定 typing_extensions 版本，否则最新版不再支持 python 3.8 导致报错
RUN pip install "typing-extensions==4.13.2" && \
    pip install /workspace/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl && \
    pip install /workspace/torchaudio-0.12.1+cu113-cp38-cp38-linux_x86_64.whl && \
    pip install torch_blade==3.27.0+1.12.1.cu113 -f https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/release/repo_ext.html && \
    pip install nvidia-tensorrt==8.4.1.5 --extra-index-url https://pypi.nvidia.com && \
    pip install hydra-core modelscope==1.26.0 huggingface-hub==0.31.2 addict datasets==2.16.0 sortedcontainers simplejson humanfriendly onnxruntime

# 6. 准备非系统包 (onnxruntime 和 ffmpeg)
COPY onnxruntime-linux-x64-gpu-1.17.1.tgz /workspace/
COPY ffmpeg-master-latest-linux64-gpl-shared.tar.xz /workspace/
RUN tar -xf ffmpeg-master-latest-linux64-gpl-shared.tar.xz && \
    tar -xzvf onnxruntime-linux-x64-gpu-1.17.1.tgz && \
    rm *.tgz *.tar.xz && \
    ln -sf /workspace/ffmpeg-master-latest-linux64-gpl-shared/bin/ffmpeg /usr/bin/ffmpeg

# 7. ⭐一定要记得把代码拷贝进镜像
COPY . /workspace/FunASR

# 7.5 安装 FunASR Python 依赖包本体
WORKDIR /workspace/FunASR
RUN pip install --no-cache-dir -e ./

# ⭐极其关键的环境变量：让 C++ 链接器能找到 Python 包里的动态库
ENV TRT_LIB_PATH="/usr/local/lib/python3.8/dist-packages/tensorrt"
ENV CUDART_LIB_PATH="/usr/local/lib/python3.8/dist-packages/torch/lib"
ENV LD_LIBRARY_PATH="${TRT_LIB_PATH}:${CUDART_LIB_PATH}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 8. 编译 C++ WebSocket Server
WORKDIR /workspace/FunASR/runtime/websocket/build
RUN cmake -DCMAKE_BUILD_TYPE=release \
      -DUSE_GPU=ON \
      -DGPU=ON \
      -DONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-gpu-1.17.1 \
      -DFFMPEG_DIR=/workspace/ffmpeg-master-latest-linux64-gpl-shared \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
      .. && \
    make -j 1

# 9. 设置运行时环境变量
ENV MODELSCOPE_CACHE=/workspace/models

# 执行命令, 进行保护 (如果你想一键启动服务，可以在这里换成 bash run_server.sh ...)
CMD ["/bin/bash"]