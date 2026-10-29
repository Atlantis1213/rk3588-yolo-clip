// Diagnostic only: input-layout negative controls and single/all-core comparison.
#include "input_io.hpp"
#include "normalization.hpp"
#include <filesystem>
#include <rknn_api.h>

int main(int argc, char** argv) {
  using namespace yolo_clip;
  rknn_context context = 0;
  bool initialized = false;
  try {
    if (argc < 5) throw std::runtime_error("probe_input_layout MODEL CORE[0|all] OUT INPUT_FP16...");
    const std::string core(argv[2]);
    if (core != "0" && core != "all") throw std::runtime_error("invalid core");
    auto check = [](int code, const char* operation) {
      if (code != RKNN_SUCC) throw std::runtime_error(std::string(operation)+": "+std::to_string(code));
    };
    auto model = ReadBinaryFile(argv[1]);
    check(rknn_init(&context,model.data(),model.size(),0,nullptr),"init");
    initialized = true;
    check(rknn_set_core_mask(context,core=="0" ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_0_1_2),"core");
    rknn_sdk_version version{};
    check(rknn_query(context,RKNN_QUERY_SDK_VERSION,&version,sizeof(version)),"version");
    rknn_tensor_attr native{},output{};
    check(rknn_query(context,RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&native,sizeof(native)),"native input");
    check(rknn_query(context,RKNN_QUERY_OUTPUT_ATTR,&output,sizeof(output)),"output");
    constexpr size_t elems = 224*224*3;
    if (native.n_elems != elems || native.fmt != RKNN_TENSOR_NHWC ||
        native.n_dims != 4 || native.dims[0]!=1 || native.dims[1]!=224 ||
        native.dims[2]!=224 || native.dims[3]!=3 || output.n_elems!=768)
      throw std::runtime_error("probe requires fixed224 NHWC native input and embedding768");
    size_t native_bytes = elems;
    if (native.type==RKNN_TENSOR_FLOAT16) native_bytes*=2;
    else if(native.type==RKNN_TENSOR_FLOAT32) native_bytes*=4;
    else if(native.type!=RKNN_TENSOR_INT8) throw std::runtime_error("unsupported native input type");
    if (native.size_with_stride!=native_bytes)
      throw std::runtime_error("native input has padding; do not guess strides");
    const std::filesystem::path destination(argv[3]);
    std::filesystem::create_directories(destination);
    std::ofstream manifest(destination/"manifest.json");
    manifest << std::setprecision(10) << "{\"core\":\"" << core << "\",\"native_type\":" << native.type
             << ",\"api_version\":\"" << JsonEscape(version.api_version)
             << "\",\"driver_version\":\"" << JsonEscape(version.drv_version) << '"'
             << ",\"native_scale\":" << native.scale << ",\"native_zp\":" << native.zp
             << ",\"model\":\"" << JsonEscape(Basename(argv[1]))
             << "\",\"skipped_modes\":\"NCHW host input rejected by runtime normalize; not a valid differential\",\"rows\":[";
    bool first = true;
    for(int index=4;index<argc;++index) {
      auto bytes = ReadBinaryFile(argv[index]);
      if(bytes.size()!=elems*2) throw std::runtime_error("unexpected packed input bytes");
      const auto* half = reinterpret_cast<const uint16_t*>(bytes.data());
      std::vector<float> hwc(elems),chw(elems);
      for(size_t pixel=0;pixel<224*224;++pixel) {
        for(size_t channel=0;channel<3;++channel) {
          const auto value = normalization::Float16ToFloat32(half[pixel*3+channel]);
          hwc[pixel*3+channel]=value;
          chw[channel*224*224+pixel]=value;
        }
      }
      std::vector<int8_t> quantized;
      if(native.type==RKNN_TENSOR_INT8) {
        if(!std::isfinite(native.scale) || native.scale<=0) throw std::runtime_error("invalid native scale");
        quantized.resize(elems);
        for(size_t i=0;i<elems;++i)
          quantized[i]=static_cast<int8_t>(std::clamp(std::nearbyint(hwc[i]/native.scale)+native.zp,-128.f,127.f));
      }
      for(const auto& mode:{std::string("hwc_nhwc"),std::string("chw_nhwc_wrong"),
                            std::string("fp16_nhwc"),std::string("native_passthrough")}) {
        rknn_input input{};
        input.type=RKNN_TENSOR_FLOAT32;
        input.fmt=RKNN_TENSOR_NHWC;
        input.buf=mode=="chw_nhwc_wrong" ? chw.data() : hwc.data();
        input.size=elems*sizeof(float);
        if(mode=="fp16_nhwc") { input.type=RKNN_TENSOR_FLOAT16; input.buf=bytes.data(); input.size=bytes.size(); }
        if(mode=="native_passthrough") {
          input.type=native.type; input.pass_through=1; input.size=native_bytes;
          if(native.type==RKNN_TENSOR_FLOAT16) input.buf=bytes.data();
          else if(native.type==RKNN_TENSOR_INT8) input.buf=quantized.data();
        }
        check(rknn_inputs_set(context,1,&input),"inputs_set");
        check(rknn_run(context,nullptr),"run");
        std::vector<float> embedding(output.n_elems);
        rknn_output result{};
        result.want_float=1; result.is_prealloc=1;
        result.buf=embedding.data(); result.size=embedding.size()*sizeof(float);
        check(rknn_outputs_get(context,1,&result,nullptr),"outputs_get");
        check(rknn_outputs_release(context,1,&result),"outputs_release");
        const auto filename=Basename(argv[index])+"."+mode+".f32";
        std::ofstream file(destination/filename,std::ios::binary);
        file.write(reinterpret_cast<const char*>(embedding.data()),embedding.size()*sizeof(float));
        if(!file) throw std::runtime_error("failed to save embedding");
        if(!first) manifest << ',';
        first=false;
        manifest << "{\"image\":\"" << JsonEscape(Basename(argv[index])) << "\",\"mode\":\"" << mode
                 << "\",\"file\":\"" << filename << "\"}";
      }
    }
    manifest << "]}\n";
    if(!manifest) throw std::runtime_error("failed to save manifest");
    rknn_destroy(context);
    std::cout << "Saved layout/core differential probe: " << destination << '\n';
  } catch(const std::exception& error) {
    if(initialized) rknn_destroy(context);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
