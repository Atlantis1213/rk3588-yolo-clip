#!/bin/sh
# 历史摄像头 demo 编译入口；第一个参数可指定实际 C++ 源码路径。
# 默认历史源码名不一定存在于当前仓库，模块化主工程请按 README 使用 CMake。

set -eu

# Keep the existing benchmark source and binary intact on the board.
g++ -std=gnu++17 -O3 -DNDEBUG -Wall -Wextra -pthread \
  -I/usr/include/rga -I/userdata/yolo_clip_bench \
  "${1:-yolo_clip_pipeline_bench.camera.cpp}" \
  -o yolo_clip_pipeline_bench_camera \
  $(pkg-config --cflags --libs opencv4) -lrga -lrknnrt
