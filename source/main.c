#include "cyhal.h"
#include "cybsp.h"

#include "mtb_ml_model.h"
#include "img_array.h"
//#include "img_array_quant.h"

/* Include model files */
#include MTB_ML_INCLUDE_MODEL_FILE(MODEL_NAME)


mtb_ml_model_t *my_model_obj;
static MTB_ML_DATA_T *result_buffer;
static int model_output_size;
volatile int a[16];
volatile int class_index;
const dtype* p_images[10];

#define DBG_1 1
#define DBG_2 2
#define QUANTIZED_INPUT 0

#if (COMPONENT_ML_INT16x16 || COMPONENT_ML_INT16x8)
    #define QFORMAT_VALUE    15
#endif
#if (COMPONENT_ML_INT8x8)
    #define QFORMAT_VALUE    7
#endif

cy_rslt_t my_model_init(void)
{
    cy_rslt_t result;

#if QUANTIZED_INPUT
    p_images[0] = img_array_quant0;
    p_images[1] = img_array_quant1;
    p_images[2] = img_array_quant2;
    p_images[3] = img_array_quant3;
    p_images[4] = img_array_quant4;
    p_images[5] = img_array_quant5;
    p_images[6] = img_array_quant6;
    p_images[7] = img_array_quant7;
    p_images[8] = img_array_quant8;
    p_images[9] = img_array_quant9;
#else
    p_images[0] = img_array0;
    p_images[1] = img_array1;
    p_images[2] = img_array2;
    p_images[3] = img_array3;
    p_images[4] = img_array4;
    p_images[5] = img_array5;
    p_images[6] = img_array6;
    p_images[7] = img_array7;
    p_images[8] = img_array8;
    p_images[9] = img_array9;
#endif	

    mtb_ml_model_bin_t my_model_bin = {MTB_ML_MODEL_BIN_DATA(MODEL_NAME)};

    /* Initialize the Neural Network */
    result = mtb_ml_model_init(&my_model_bin, NULL, &my_model_obj);
    if(CY_RSLT_SUCCESS != result)
    {
		a[1]=DBG_1;
        return result;
    }

#if !COMPONENT_ML_FLOAT32
    /* Set the q-factor */
    mtb_ml_model_set_input_q_fraction_bits(my_model_obj, QFORMAT_VALUE);
#endif

    mtb_ml_model_get_output(my_model_obj, &result_buffer, &model_output_size);

    return result;
}

int control(MTB_ML_DATA_T* result_buffer, int model_output_size)
{
    /* Get the class with the highest confidence */
    int class_index = mtb_ml_utils_find_max(result_buffer, model_output_size);

#if !COMPONENT_ML_FLOAT32
    /* Convert 16bit fixed-point output to floating-point for visualization */
    float *nn_float_buffer = (float *) malloc(model_output_size * sizeof(float));
    mtb_ml_utils_model_dequantize(my_model_obj, nn_float_buffer);
#else
    float *nn_float_buffer = result_buffer;
#endif

    free(nn_float_buffer);
	return class_index;
}

void quantize_input(mtb_ml_model_t *my_model_obj, float* src, int8_t* dest)
{
	for (int i=0; i<my_model_obj->input_size; i++) {
 	 float x = src[i];
     int8_t x_quantized = x / my_model_obj->input_scale + my_model_obj->input_zero_point;
     dest[i] = x_quantized;
	}
}

int main(void)
{
    cy_rslt_t result;
    MTB_ML_DATA_T *input_reference;
    int8_t p_image_int[28*28]; // static memory for 28*28 grayscaled image

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

	// Run inference: idx=4
	for (int i=4; i<5; i++) {
		
#if QUANTIZED_INPUT
        /* Feed the Model */
        input_reference = (MTB_ML_DATA_T *) p_images[i];
        mtb_ml_model_run(my_model_obj, input_reference);
        class_index = control(result_buffer, model_output_size);

#elif !COMPONENT_ML_FLOAT32
        /* Quantize data before feeding model */
        //mtb_ml_utils_model_quantize(my_model_obj, p_images[i], p_image_int);
        quantize_input(my_model_obj, p_images[i], p_image_int);

        /* Feed the Model */
        input_reference = (MTB_ML_DATA_T *) p_image_int;
        mtb_ml_model_run(my_model_obj, input_reference);
        class_index = control(result_buffer, model_output_size);
#else
        input_reference = (MTB_ML_DATA_T *)p_images[i];
        mtb_ml_model_run(my_model_obj, input_reference);
        class_index = control(result_buffer, model_output_size); 
#endif
	 
 	    a[i]=class_index;
	}
	
    a[11]=class_index;
	a[12]=DBG_2;

    for (;;)
    {
    }
}

/* [] END OF FILE */
