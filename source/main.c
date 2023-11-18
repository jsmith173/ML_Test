#include "cyhal.h"
#include "cybsp.h"

#include "main_functions.h"


static int model_output_size;
volatile int a[16];
volatile int class_index;

#define DBG_1 1
#define DBG_2 2

int main(void)
{
    cy_rslt_t result;

    /* Initialize the device and board peripherals */
    result = cybsp_init();

    /* Board init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    ai_setup();
    ai_loop();
	
    /* Enable global interrupts */
    __enable_irq();

	
    a[11]=class_index;
	a[12]=DBG_2;

    for (;;)
    {
    }
}

/* [] END OF FILE */
