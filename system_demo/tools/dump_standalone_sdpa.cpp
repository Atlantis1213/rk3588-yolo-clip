// Diagnostic-only three-input SDPA, CPU host submission with checked native geometry.
#include "input_io.hpp"
#include <filesystem>
#include <rknn_api.h>

int main(int argc, char** argv) {
  using namespace yolo_clip;
  rknn_context context = 0;
  bool initialized = false;
  try {
    if (argc != 4) {
      throw std::runtime_error("dump_standalone_sdpa MODEL INPUT_DIR OUTPUT_DIR");
    }
    auto check = [](int code, const char* op) {
      if (code != RKNN_SUCC) {
        throw std::runtime_error(std::string(op) + ": " + std::to_string(code));
      }
    };
    auto model = ReadBinaryFile(argv[1]);
    check(rknn_init(&context, model.data(), model.size(), RKNN_FLAG_COLLECT_PERF_MASK, nullptr),
          "init");
    initialized = true;
    check(rknn_set_core_mask(context, RKNN_NPU_CORE_0), "core");
    rknn_input_output_num count{};
    check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)), "count");
    if (count.n_input != 3 || count.n_output != 1) {
      throw std::runtime_error("unexpected IO count");
    }
    for (uint32_t i = 0; i < 3; ++i) {
      rknn_tensor_attr attr{};
      attr.index = i;
      // Host submission is converted by RKNN; native storage may be NC1HWC2.
      check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "logical input attr");
      if (attr.n_dims != 4 || attr.dims[0] != 1 || attr.dims[1] != 257 || attr.dims[2] != 64 ||
          attr.dims[3] != 16) {
        throw std::runtime_error("unhandled native layout");
      }
    }
    rknn_tensor_attr output{};
    check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)), "output");
    if (output.n_elems != 16 * 257 * 64) {
      throw std::runtime_error("unexpected output size");
    }
    const std::filesystem::path inputs(argv[2]), destination(argv[3]);
    std::filesystem::create_directories(destination);
    std::ofstream metadata(destination / "manifest.json");
    metadata << "{\"dims\":[";
    for (uint32_t d = 0; d < output.n_dims; ++d) {
      if (d) {
        metadata << ',';
      }
      metadata << output.dims[d];
    }
    metadata << "]}\n";
    bool first = true;
    for (const auto& name : {std::string("cardboard"),
                             std::string("glass"),
                             std::string("metal"),
                             std::string("paper"),
                             std::string("plastic"),
                             std::string("trash")}) {
      std::vector<std::vector<uint8_t>> buffers;
      for (const auto& key : {"q", "k", "v"}) {
        buffers.push_back(ReadBinaryFile((inputs / (name + "_" + key + ".fp16")).string()));
      }
      rknn_input requests[3]{};
      for (uint32_t i = 0; i < 3; ++i) {
        if (buffers[i].size() != 16 * 257 * 64 * 2) {
          throw std::runtime_error("invalid input size");
        }
        requests[i].index = i;
        requests[i].fmt = RKNN_TENSOR_NHWC;
        requests[i].type = RKNN_TENSOR_FLOAT16;
        requests[i].buf = buffers[i].data();
        requests[i].size = buffers[i].size();
      }
      check(rknn_inputs_set(context, 3, requests), "input");
      check(rknn_run(context, nullptr), "run");
      std::vector<float> result(output.n_elems);
      rknn_output response{};
      response.want_float = 1;
      response.is_prealloc = 1;
      response.buf = result.data();
      response.size = result.size() * sizeof(float);
      check(rknn_outputs_get(context, 1, &response, nullptr), "get");
      check(rknn_outputs_release(context, 1, &response), "release");
      std::ofstream file(destination / (name + ".f32"), std::ios::binary);
      file.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(float));
      if (!file) {
        throw std::runtime_error("save failed");
      }
      if (first) {
        rknn_perf_detail detail{};
        check(rknn_query(context, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)), "perf");
        std::ofstream perf(destination / "perf_detail.txt");
        perf.write(detail.perf_data, detail.data_len);
        first = false;
      }
    }
    rknn_destroy(context);
    std::cout << "Saved pure SDPA result " << destination << '\n';
  }
  catch (const std::exception& error) {
    if (initialized) {
      rknn_destroy(context);
    }
    std::cerr << error.what() << '\n';
    return 1;
  }
}
