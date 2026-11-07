#!/bin/sh
# 根目录历史版本的板端原生编译脚本，需本机安装 OpenCV/RGA/RKNN 开发库。
# 脚本依赖历史源文件 yolo_clip_pipeline_bench.cpp；当前模块化主工程请使用 system_demo/CMakeLists.txt。

set -eu

g++ -std=gnu++17 -O3 -DNDEBUG -Wall -Wextra -pthread \
  -I/usr/include/rga -I/userdata/yolo_clip_bench \
  yolo_clip_pipeline_bench.cpp \
  -o yolo_clip_pipeline_bench_optimized \
  $(pkg-config --cflags --libs opencv4) -lrga -lrknnrt
