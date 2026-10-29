#pragma once
#include "model_runner.hpp"
#include "pipeline_queue.hpp"
#include "metrics.hpp"
#include "text_bank.hpp"
#include "scheduler.hpp"

namespace yolo_clip {
Options ParseArgs(int, char**);
void RunSerial(const Options&, const std::vector<cv::Mat>&, const std::vector<std::string>&, const TextBank&, const std::vector<uint8_t>&, const std::vector<uint8_t>&);
void RunPool(const Options&, const std::vector<cv::Mat>&, const std::vector<std::string>&, const TextBank&, const std::vector<uint8_t>&, const std::vector<uint8_t>&);
}
