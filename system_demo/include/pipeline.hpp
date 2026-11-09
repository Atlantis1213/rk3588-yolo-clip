// 对 main 暴露的流水线接口：统一参数解析，串行基线和线程池接受相同模型/文本库。
// 具体生产者、worker 与退出协议见 src/pipeline.cpp。

#pragma once
#include "model_runner.hpp"
#include "pipeline_queue.hpp"
#include "metrics.hpp"
#include "text_bank.hpp"
#include "scheduler.hpp"

namespace yolo_clip {
Options ParseArgs(int, char**);
void RunSerial(const Options&,
               const std::vector<cv::Mat>&,
               const std::vector<std::string>&,
               const TextBank&,
               const std::vector<uint8_t>&,
               const std::vector<uint8_t>&);
void RunPool(const Options&,
             const std::vector<cv::Mat>&,
             const std::vector<std::string>&,
             const TextBank&,
             const std::vector<uint8_t>&,
             const std::vector<uint8_t>&);
}  // namespace yolo_clip
