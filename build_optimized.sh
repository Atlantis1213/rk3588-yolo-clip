#!/bin/sh
set -eu

g++ -std=gnu++17 -O3 -DNDEBUG -Wall -Wextra -pthread \
  -I/usr/include/rga -I/userdata/yolo_clip_bench \
  yolo_clip_pipeline_bench.cpp \
  -o yolo_clip_pipeline_bench_optimized \
  $(pkg-config --cflags --libs opencv4) -lrga -lrknnrt
