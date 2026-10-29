#!/bin/sh
set -eu

# Keep the existing benchmark source and binary intact on the board.
g++ -std=gnu++17 -O3 -DNDEBUG -Wall -Wextra -pthread \
  -I/usr/include/rga -I/userdata/yolo_clip_bench \
  "${1:-yolo_clip_pipeline_bench.camera.cpp}" \
  -o yolo_clip_pipeline_bench_camera \
  $(pkg-config --cflags --libs opencv4) -lrga -lrknnrt
