#include "input_io.hpp"
#include <rknn_api.h>

int main(int argc, char** argv) {
  using namespace yolo_clip;
  rknn_context context = 0;
  bool initialized = false;
  try {
    if (argc != 4) throw std::runtime_error("bench_clip_batch MODEL INPUT_FP16 DURATION");
    const auto model = ReadBinaryFile(argv[1]), frame = ReadBinaryFile(argv[2]);
    if (frame.size() != 224 * 224 * 3 * 2) throw std::runtime_error("invalid single-frame tensor");
    int status = rknn_init(&context, const_cast<uint8_t*>(model.data()), model.size(), 0, nullptr);
    if (status) throw std::runtime_error("batch init failed");
    initialized = true;
    rknn_set_core_mask(context, RKNN_NPU_CORE_0_1_2);
    rknn_tensor_attr input_attr{}, output_attr{};
    rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    const int batch = input_attr.n_elems / (224 * 224 * 3);
    if (batch <= 0 || input_attr.n_elems != static_cast<uint32_t>(batch * 224 * 224 * 3))
      throw std::runtime_error("unexpected batch input shape");
    std::vector<uint8_t> input(frame.size() * batch);
    for (int i = 0; i < batch; ++i) std::memcpy(input.data() + i * frame.size(), frame.data(), frame.size());
    std::vector<float> output(output_attr.n_elems);
    auto infer = [&] {
      rknn_input in{}; in.index = 0; in.type = RKNN_TENSOR_FLOAT16; in.fmt = RKNN_TENSOR_NHWC;
      in.buf = input.data(); in.size = input.size();
      if (rknn_inputs_set(context, 1, &in)) throw std::runtime_error("batch input set failed");
      rknn_output out{}; out.want_float = 1; out.is_prealloc = 1; out.buf = output.data(); out.size = output.size() * 4;
      const auto start = Clock::now();
      if (rknn_run(context, nullptr) || rknn_outputs_get(context, 1, &out, nullptr))
        throw std::runtime_error("batch inference failed");
      const auto milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      rknn_outputs_release(context, 1, &out);
      return milliseconds;
    };
    for (int i = 0; i < 2; ++i) infer();
    const auto start = Clock::now(), deadline = start + std::chrono::seconds(std::stoi(argv[3]));
    int iterations = 0; double sum = 0;
    while (Clock::now() < deadline) { sum += infer(); ++iterations; }
    const double wall = std::chrono::duration<double>(Clock::now() - start).count();
    const size_t dimension = output.size() / batch;
    double maximum_difference = 0;
    for (int i = 1; i < batch; ++i) for (size_t k = 0; k < dimension; ++k)
      maximum_difference = std::max(maximum_difference, static_cast<double>(std::abs(output[k] - output[i * dimension + k])));
    std::cout << "{\"batch\":" << batch << ",\"iterations\":" << iterations
              << ",\"frames\":" << iterations * batch << ",\"fps\":" << iterations * batch / wall
              << ",\"batch_run_ms_mean\":" << sum / iterations
              << ",\"identical_input_max_output_diff\":" << maximum_difference << "}\n";
    rknn_destroy(context);
  } catch (const std::exception& error) { if (initialized) rknn_destroy(context); std::cerr << error.what() << '\n'; return 1; }
}
