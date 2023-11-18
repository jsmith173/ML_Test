//#define USE_FLOAT

#include "main_functions.h"
#include "img_array.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "model_quant_data.h"

#define ARENA_SIZE (440*1024)

namespace tflite {

static tflite::AllOpsResolver resolver;

template <typename inputT>
class MTBTFLiteMicro {
 public:
  // The lifetimes of model, op_resolver, tensor_arena must exceed
  // that of the created MicroBenchmarkRunner object.
  MTBTFLiteMicro(const uint8_t* model,
                       uint8_t* tensor_arena, int tensor_arena_size,
                       const tflite::MicroOpResolver& op_resolver)
      : interpreter_(GetModel(model), op_resolver, tensor_arena,
                     tensor_arena_size, GetMicroErrorReporter(), nullptr, nullptr) {
      allocate_status_ = interpreter_.AllocateTensors();
  }

  TfLiteStatus RunSingleIteration() {
    // Run the model on this input and return the status.
    return interpreter_.Invoke();
  }

  TfLiteTensor* Input(int index = 0)  { return interpreter_.input(index); }
  TfLiteTensor* Output(int index = 0) { return interpreter_.output(index); }

  TfLiteStatus AllocationStatus() { return allocate_status_; }

  inputT* input_ptr(int index = 0) { return GetTensorData<inputT>(Input(index)); }
  size_t input_size(int index = 0) { return interpreter_.input(index)->bytes; }
  size_t input_elements(int index = 0) { return tflite::ElementCount(*(interpreter_.input(index)->dims)); }
  int    input_dims_len(int index=0) {return interpreter_.input(index)->dims->size; }
  int *  input_dims( int index=0) { return &interpreter_.input(index)->dims->data[0]; }
  int    input_zero_point( int index=0) { return interpreter_.input(index)->params.zero_point; }
  float  input_scale( int index=0) { return interpreter_.input(index)->params.scale; }
  int    output_zero_point( int index=0) { return interpreter_.output(index)->params.zero_point; }
  float  output_scale( int index=0) { return interpreter_.output(index)->params.scale; }

  inputT* output_ptr(int index = 0) { return GetTensorData<inputT>(Output(index)); }
  size_t output_size(int index = 0) { return interpreter_.output(index)->bytes; }
  size_t output_elements(int index = 0) { return  tflite::ElementCount(*(interpreter_.output(index)->dims));}
  int    output_dims_len(int index=0) {return interpreter_.output(index)->dims->size; }
  int *  output_dims( int index=0) { return &interpreter_.output(index)->dims->data[0]; }
  size_t get_used_arena_size() { return (interpreter_.arena_used_bytes() + 1023); }

  void SetInput(const inputT* custom_input, int input_index = 0) {
    TfLiteTensor* input = interpreter_.input(input_index);
    inputT* input_buffer = tflite::GetTensorData<inputT>(input);
    int input_length = input->bytes / sizeof(inputT);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = custom_input[i];
    }
  }

  void PrintAllocations() const {
    interpreter_.GetMicroAllocator().PrintAllocations();
  }

 private:
  tflite::RecordingMicroInterpreter interpreter_;
  TfLiteStatus allocate_status_;
};

using MTB_TFLM_flt = MTBTFLiteMicro<float>;
using MTB_TFLM_int8 = MTBTFLiteMicro<int8_t>;

} // namespace tflite

#ifdef COMPONENT_ML_INT8x8
#define MTB_TFLM_Class MTB_TFLM_int8
#else
#define MTB_TFLM_Class MTB_TFLM_flt
#endif

namespace {
 TfLiteTensor* input = nullptr;
 TfLiteTensor* output = nullptr;

 constexpr int kTensorArenaSize = ARENA_SIZE;
 uint8_t tensor_arena[kTensorArenaSize];
 tflite::MTB_TFLM_Class * TFLMClass;
}  // namespace


template <typename T>
int argmax() 
{
  int size = output->bytes; 
  int div_size = 1;
  if (output->type == kTfLiteFloat32)
    div_size = sizeof(float);
  int num_classes = size/div_size;
  
  int idx = 0;
  T vi, v = 0;
  for (uint32_t i = 0; i < num_classes; i++) {
    T* output_data = tflite::GetTensorData<T>(output);
    for (int i = 0; i < num_classes; i++) {
      vi = output_data[i];
      if (vi > v){
        idx = i;
        v = vi;
      }
    }
  }
  return idx;
}

// The name of this function is important for Arduino compatibility.
int ai_setup() 
{
  TFLMClass = new tflite::MTB_TFLM_Class(g_model_quant_data, tensor_arena, kTensorArenaSize, tflite::resolver);
  
  input = TFLMClass->Input();
  output = TFLMClass->Output();
  
  return 1;
}

int ai_loop()
{
  int tensor_size = input->bytes;
  int dim_size = input->dims->size;
  int dim_1 = input->dims->data[0];
  int dim_h = input->dims->data[1];
  int dim_w = input->dims->data[2];
  int dim_pic = 1;
  if (dim_size == 4)
   dim_pic = input->dims->data[3];
  int N = dim_w*dim_h;
  int passed, data, k;
  uint8_t ui8Data;
  
}
