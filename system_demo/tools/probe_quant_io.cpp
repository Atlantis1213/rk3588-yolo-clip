// Diagnostic only: separate host conversion, native pass-through, and output dequantization.
#include "input_io.hpp"
#include "normalization.hpp"
#include <filesystem>
#include <rknn_api.h>

int main(int argc, char** argv) {
  using namespace yolo_clip;
  rknn_context context = 0;
  bool initialized = false;
  try {
    if (argc < 4) throw std::runtime_error("probe_quant_io MODEL OUTPUT_DIR INPUT_FP16 [INPUT_FP16 ...]");
    auto model = ReadBinaryFile(argv[1]);
    auto check = [](int code, const char* operation) {
      if (code != RKNN_SUCC) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
    };
    check(rknn_init(&context, model.data(), model.size(), 0, nullptr), "rknn_init");
    initialized = true;
    check(rknn_set_core_mask(context, RKNN_NPU_CORE_0_1_2), "core mask");
    rknn_tensor_attr input{}, native{}, output{};
    check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)), "input query");
    check(rknn_query(context, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &native, sizeof(native)), "native query");
    check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)), "output query");
    if (native.type != RKNN_TENSOR_INT8 || native.fmt != RKNN_TENSOR_NHWC ||
        native.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || native.scale <= 0 ||
        native.n_dims != 4 || native.dims[0] != 1 || native.dims[1] != 224 ||
        native.dims[2] != 224 || native.dims[3] != 3 || native.size_with_stride != native.n_elems ||
        output.type != RKNN_TENSOR_INT8 || output.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
        output.scale <= 0)
      throw std::runtime_error("probe requires packed NHWC affine INT8 input/output");
    const std::filesystem::path destination(argv[2]);
    std::filesystem::create_directories(destination);
    std::ofstream manifest(destination / "manifest.json");
    manifest << std::setprecision(10) << "{\"input_scale\":" << native.scale << ",\"input_zp\":" << native.zp
             << ",\"output_scale\":" << output.scale << ",\"output_zp\":" << output.zp
             << ",\"embedding_dim\":" << output.n_elems << ",\"rows\":[";
    bool first = true;
    for (int index = 3; index < argc; ++index) {
      auto half_bytes = ReadBinaryFile(argv[index]);
      if (half_bytes.size() != native.n_elems * 2) throw std::runtime_error("input tensor size mismatch");
      std::vector<float> floats(native.n_elems);
      std::vector<int8_t> quantized(native.n_elems);
      const auto* halves = reinterpret_cast<const uint16_t*>(half_bytes.data());
      for (size_t i = 0; i < floats.size(); ++i) {
        floats[i] = normalization::Float16ToFloat32(halves[i]);
        const float q = std::nearbyint(floats[i] / native.scale) + native.zp;
        quantized[i] = static_cast<int8_t>(std::clamp(q, -128.f, 127.f));
      }
      for (const auto& mode : {std::string("fp16_host"), std::string("fp32_host"),
                               std::string("int8_host"), std::string("int8_passthrough")}) {
        rknn_input request{};
        request.fmt = RKNN_TENSOR_NHWC;
        if (mode == "fp16_host") {
          request.type = RKNN_TENSOR_FLOAT16; request.buf = half_bytes.data(); request.size = half_bytes.size();
        } else if (mode == "fp32_host") {
          request.type = RKNN_TENSOR_FLOAT32; request.buf = floats.data(); request.size = floats.size() * sizeof(float);
        } else {
          request.type = RKNN_TENSOR_INT8; request.buf = quantized.data(); request.size = quantized.size();
          request.pass_through = mode == "int8_passthrough";
        }
        std::vector<float> actual(output.n_elems);
        std::vector<int8_t> raw(output.n_elems);
        for (int pass = 0; pass < 2; ++pass) {
          check(rknn_inputs_set(context, 1, &request), "inputs_set");
          check(rknn_run(context, nullptr), "run");
          rknn_output result{};
          result.want_float = pass == 0; result.is_prealloc = 1;
          result.buf = pass == 0 ? static_cast<void*>(actual.data()) : static_cast<void*>(raw.data());
          result.size = pass == 0 ? actual.size() * sizeof(float) : raw.size();
          check(rknn_outputs_get(context, 1, &result, nullptr), "outputs_get");
          check(rknn_outputs_release(context, 1, &result), "outputs_release");
        }
        double max_error = 0;
        for (size_t i = 0; i < raw.size(); ++i)
          max_error = std::max(max_error, static_cast<double>(std::abs(actual[i] - (raw[i] - output.zp) * output.scale)));
        const auto filename = Basename(argv[index]) + "." + mode + ".f32";
        std::ofstream file(destination / filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(actual.data()), actual.size() * sizeof(float));
        if (!file) throw std::runtime_error("failed to write embedding");
        if (!first) manifest << ',';
        first = false;
        manifest << "{\"image\":\"" << JsonEscape(Basename(argv[index])) << "\",\"mode\":\"" << mode
                 << "\",\"file\":\"" << filename << "\",\"output_dequant_max_abs_error\":" << max_error << '}';
      }
    }
    manifest << "]}\n";
    if (!manifest) throw std::runtime_error("failed to write manifest");
    rknn_destroy(context);
    std::cout << "Quantization IO probe saved to " << destination << '\n';
  } catch (const std::exception& error) {
    if (initialized) rknn_destroy(context);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
