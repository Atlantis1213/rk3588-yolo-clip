// Diagnostic-only intermediate output dump; no change to deployed model.
#include "input_io.hpp"
#include <filesystem>
#include <rknn_api.h>

int main(int argc,char** argv) {
  using namespace yolo_clip;
  rknn_context context=0;
  bool initialized=false;
  try {
    if(argc<4) throw std::runtime_error("dump_prefix_outputs MODEL OUT INPUT_FP16...");
    auto check=[](int code,const char* op){if(code!=RKNN_SUCC) throw std::runtime_error(std::string(op)+": "+std::to_string(code));};
    const bool perf=std::string(argv[3])=="--perf";
    const int first_input=perf ? 4 : 3;
    if(first_input>=argc) throw std::runtime_error("missing input tensor");
    auto bytes=ReadBinaryFile(argv[1]);
    check(rknn_init(&context,bytes.data(),bytes.size(),perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0,nullptr),"init"); initialized=true;
    check(rknn_set_core_mask(context,RKNN_NPU_CORE_0),"core");
    rknn_tensor_attr output{};
    check(rknn_query(context,RKNN_QUERY_OUTPUT_ATTR,&output,sizeof(output)),"output");
    if(output.n_elems==0 || output.n_elems>1024*1024) throw std::runtime_error("unexpected output size");
    const std::filesystem::path destination(argv[2]);
    std::filesystem::create_directories(destination);
    std::ofstream manifest(destination/"manifest.json");
    manifest << "{\"n_elems\":" << output.n_elems << ",\"fmt\":" << output.fmt << ",\"dims\":[";
    for(uint32_t d=0;d<output.n_dims;++d) { if(d) manifest << ','; manifest << output.dims[d]; }
    manifest << "],\"rows\":[";
    for(int index=first_input;index<argc;++index) {
      auto input=ReadBinaryFile(argv[index]);
      if(input.size()!=224*224*3*2) throw std::runtime_error("invalid packed NHWC half input");
      rknn_input request{}; request.fmt=RKNN_TENSOR_NHWC; request.type=RKNN_TENSOR_FLOAT16;
      request.buf=input.data(); request.size=input.size();
      check(rknn_inputs_set(context,1,&request),"input"); check(rknn_run(context,nullptr),"run");
      std::vector<float> result(output.n_elems);
      rknn_output response{}; response.want_float=1; response.is_prealloc=1;
      response.buf=result.data(); response.size=result.size()*sizeof(float);
      check(rknn_outputs_get(context,1,&response,nullptr),"get"); check(rknn_outputs_release(context,1,&response),"release");
      if(perf && index==first_input) {
        rknn_perf_detail detail{};
        const int code=rknn_query(context,RKNN_QUERY_PERF_DETAIL,&detail,sizeof(detail));
        std::ofstream report(destination/"perf_detail.txt");
        if(code==RKNN_SUCC && detail.perf_data) report.write(detail.perf_data,detail.data_len);
        else report << "query failed: " << code << '\n';
        if(!report) throw std::runtime_error("cannot save diagnostic performance detail");
      }
      const auto filename=Basename(argv[index])+".f32";
      std::ofstream stream(destination/filename,std::ios::binary);
      stream.write(reinterpret_cast<const char*>(result.data()),response.size);
      if(!stream) throw std::runtime_error("cannot save output");
      if(index>first_input) manifest << ',';
      manifest << "{\"image\":\"" << JsonEscape(Basename(argv[index])) << "\",\"file\":\"" << filename << "\"}";
    }
    manifest << "]}\n"; if(!manifest) throw std::runtime_error("cannot save manifest");
    rknn_destroy(context);
    std::cout << "Saved prefix outputs: " << destination << '\n';
  } catch(const std::exception& error) {
    if(initialized) rknn_destroy(context);
    std::cerr << error.what() << '\n'; return 1;
  }
}
