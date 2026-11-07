// 硬件主工程的公共依赖入口。可移植队列/缓冲池/归一化头文件刻意不依赖此文件。
// Clock 使用单调时钟，性能计时不受系统时间校准影响。

#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <im2d.h>
#include <rga/im2d_buffer.h>

#include "rknn_api.h"
#include "camera_dma.hpp"


namespace yolo_clip { using Clock = std::chrono::steady_clock; }
