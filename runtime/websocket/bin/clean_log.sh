#!/bin/bash

log_dir="/workspace/FunASR/logs"
max_files=21

# 删除最旧的日志文件，只保留最新的 max_files 个日志文件
ls -1t $log_dir/*.log* | tail -n +$((max_files + 1)) | xargs rm -f