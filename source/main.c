#include "cyhal.h"
#include "cybsp.h"

#include "mtb_ml_model.h"

/* Include model files */
#include MTB_ML_INCLUDE_MODEL_FILE(MODEL_NAME)

extern const float img_array[];

mtb_ml_model_t *image_class_obj;
static MTB_ML_DATA_T *result_buffer;
static int model_output_size;
volatile int a[16];
volatile float f1;
volatile int class_index;

#define DBG_1 1
#define DBG_2 2

cy_rslt_t image_class_init(void)
{
    cy_rslt_t result;

    mtb_ml_model_bin_t image_class_bin = {MTB_ML_MODEL_BIN_DATA(MODEL_NAME)};
	f1 = img_array[0];

    /* Initialize the Neural Network */
    result = mtb_ml_model_init(&image_class_bin, NULL, &image_class_obj);
    if(CY_RSLT_SUCCESS != result)
    {
		a[1]=DBG_1;
        return result;
    }

#if !COMPONENT_ML_FLOAT32
    /* Set the q-factor */
    mtb_ml_model_set_input_q_fraction_bits(image_class_obj, QFORMAT_VALUE);
#endif

    mtb_ml_model_get_output(image_class_obj, &result_buffer, &model_output_size);

    return result;
}

int control(MTB_ML_DATA_T* result_buffer, int model_output_size)
{
    /* Get the class with the highest confidence */
    int class_index = mtb_ml_utils_find_max(result_buffer, model_output_size);

#if !COMPONENT_ML_FLOAT32
    /* Convert 16bit fixed-point output to floating-point for visualization */
    float *nn_float_buffer = (float *) malloc(model_output_size * sizeof(float));
    mtb_ml_utils_model_dequantize(magic_wand_obj, nn_float_buffer);
#else
    float *nn_float_buffer = result_buffer;
#endif

    free(nn_float_buffer);
	return class_index;
}

int main(void)
{
    cy_rslt_t result;
    MTB_ML_DATA_T *input_reference;

    /* Initialize the device and board peripherals */
    result = cybsp_init();
    result = image_class_init();

    /* Board init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    /* Enable global interrupts */
    __enable_irq();

	// Run inference
    input_reference = (MTB_ML_DATA_T *) img_array;
    mtb_ml_model_run(image_class_obj, input_reference);
    class_index = control(result_buffer, model_output_size);
	
	a[1]=DBG_2;
	
    for (;;)
    {
    }
}

/* [] END OF FILE */
