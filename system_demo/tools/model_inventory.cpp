#include "rknn_api.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

struct Context {
  rknn_context value = 0;
  bool initialized = false;
  ~Context() { if (initialized) rknn_destroy(value); }
};
void Check(int code, const char* operation) {
  if (code != RKNN_SUCC) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}
void PrintTensor(const rknn_tensor_attr& attr) {
  std::cout << "{\"type_enum\":" << attr.type << ",\"format_enum\":" << attr.fmt
            << ",\"quantization_enum\":" << attr.qnt_type << ",\"elements\":" << attr.n_elems
            << ",\"bytes\":" << attr.size << ",\"bytes_with_stride\":" << attr.size_with_stride
            << ",\"width_stride\":" << attr.w_stride << ",\"scale\":" << std::setprecision(12) << attr.scale
            << ",\"zero_point\":" << attr.zp << ",\"dims\":[";
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (i) std::cout << ',';
    std::cout << attr.dims[i];
  }
  std::cout << "]}";
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("model_inventory YOLO.rknn CLIP.rknn");
    std::cout << '[';
    for (int i = 1; i < argc; ++i) {
      std::ifstream file(argv[i], std::ios::binary | std::ios::ate);
      if (!file) throw std::runtime_error("cannot open model");
      const auto size = static_cast<size_t>(file.tellg());
      file.seekg(0);
      std::vector<uint8_t> bytes(size);
      file.read(reinterpret_cast<char*>(bytes.data()), size);
      Context context;
      Check(rknn_init(&context.value, bytes.data(), bytes.size(), 0, nullptr), "initialize model");
      context.initialized = true;
      rknn_sdk_version version{};
      Check(rknn_query(context.value, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)), "query version");
      rknn_tensor_attr input{}, native{}, output{};
      Check(rknn_query(context.value, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)), "query input");
      Check(rknn_query(context.value, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &native, sizeof(native)), "query native input");
      Check(rknn_query(context.value, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)), "query output");
      if (i > 1) std::cout << ',';
      std::cout << "\n{\"role\":\"" << (i == 1 ? "yolo" : "clip") << "\",\"model_bytes\":" << size
                << ",\"runtime\":\"" << version.api_version << "\",\"driver\":\"" << version.drv_version
                << "\",\"input\":";
      PrintTensor(input); std::cout << ",\"native_input\":"; PrintTensor(native);
      std::cout << ",\"output\":"; PrintTensor(output); std::cout << '}';
    }
    std::cout << "\n]\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
