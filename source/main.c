#include "cyhal.h"
#include "cybsp.h"

#include "mtb_ml_model.h"
#include "img_array.h"

/* Include model files */


mtb_ml_model_t *my_model_obj;
static MTB_ML_DATA_T *result_buffer;
static int model_output_size;
volatile int a[16];
volatile int class_index;
const dtype* p_image;

#define DBG_1 1
#define DBG_2 2

#define ARENA_SIZE (440*1024)

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
 const tflite::Model* model = nullptr;
 tflite::MicroInterpreter* interpreter = nullptr;
 TfLiteTensor* input = nullptr;
 TfLiteTensor* output = nullptr;

 constexpr int kTensorArenaSize = ARENA_SIZE;
 uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

cy_rslt_t my_model_init(void)
{
    cy_rslt_t result;

    p_image = img_array;

    /* Initialize the Neural Network */
    TFLMClass = new tflite::MTB_TFLM_Class(bin->model_bin, tensor_arena, kTensorArenaSize, tflite::resolver);

    return CY_RSLT_SUCCESS;
}

int main(void)
{
    cy_rslt_t result;
    MTB_ML_DATA_T *input_reference;
    int8_t p_image_int[96*96*3]; // static memory for 96*96*3 image

    /* Initialize the device and board peripherals */
    result = cybsp_init();

    /* Board init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    my_model_init();
	
    /* Enable global interrupts */
    __enable_irq();

	
    a[11]=class_index;
	a[12]=DBG_2;

    for (;;)
    {
    }
}

/* [] END OF FILE */
