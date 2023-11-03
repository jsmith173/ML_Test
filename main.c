#include "cyhal.h"
#include "cybsp.h"

#include "mtb_ml_model.h"

/* Include model files */
#include MTB_ML_INCLUDE_MODEL_FILE(MODEL_NAME)

mtb_ml_model_t *image_class_obj;
static MTB_ML_DATA_T *result_buffer;
static int model_output_size;


cy_rslt_t image_class_init(void)
{
    cy_rslt_t result;

    mtb_ml_model_bin_t image_class_bin = {MTB_ML_MODEL_BIN_DATA(MODEL_NAME)};

    /* Initialize the Neural Network */
    result = mtb_ml_model_init(&image_class_bin, NULL, &image_class_obj);
    if(CY_RSLT_SUCCESS != result)
    {
        return result;
    }

#if !COMPONENT_ML_FLOAT32
    /* Set the q-factor */
    mtb_ml_model_set_input_q_fraction_bits(image_class_obj, QFORMAT_VALUE);
#endif

    mtb_ml_model_get_output(image_class_obj, &result_buffer, &model_output_size);

    return result;
}

int main(void)
{
    cy_rslt_t result;

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

    for (;;)
    {
    }
}

/* [] END OF FILE */
