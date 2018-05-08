/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
 
 /*
 * This sample implements a separable convolution filter 
 * of a 2D image with an arbitrary kernel.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil_inline.h>

#include "../benchmark_common.h"

#include "convolutionSeparable_common.h"



////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);




////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv){
int main_CONS(cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag){
    float
        *h_Kernel,
        *h_Input,
        *h_Buffer,
        *h_OutputCPU,
        *h_OutputGPU;

    float
        *d_Input,
        *d_Output,
        *d_Buffer;


    const int imageW = 3072;
    const int imageH = 3072 / 2;
    const unsigned int iterations = 10;

    unsigned int hTimer;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    /*if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device") )
        cutilDeviceInit(argc, argv);
    else*/
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError(cutCreateTimer(&hTimer));

    printf("%i x %i\n", imageW, imageH);
    printf("Allocating and intializing host arrays...\n");
        h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
        h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
        h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
        h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
        h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
        srand(200);
        for(unsigned int i = 0; i < KERNEL_LENGTH; i++)
            h_Kernel[i] = (float)(rand() % 16);
        for(unsigned i = 0; i < imageW * imageH; i++)
            h_Input[i] = (float)(rand() % 16);


    printf("Allocating and initializing CUDA arrays...\n");
        cutilSafeCall( cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)) );
        cutilSafeCall( cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)) );
        cutilSafeCall( cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float)) );

        setConvolutionKernel(h_Kernel, stream_app);
        cutilSafeCall( cudaMemcpyAsync(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice, stream_app) );


    printf("Running GPU convolution (%u identical iterations)...\n", iterations);
        //cutilSafeCall( cudaThreadSynchronize() );
        if(flag)
            cutilSafeCall( cudaStreamSynchronize(stream_app) );
        else
            cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutResetTimer(hTimer) );
        cutilCheckError( cutStartTimer(hTimer) );
            for(unsigned int i = 0; i < iterations; i++){
                convolutionRowsGPU(
                    d_Buffer,
                    d_Input,
                    imageW,
                    imageH, stream_app
                );
					
                convolutionColumnsGPU(
                    d_Output,
                    d_Buffer,
                    imageW,
                    imageH,stream_app
                );
            }
        //cutilSafeCall( cudaThreadSynchronize() );
        pthread_mutex_unlock (mutexapp);
    if(flag)
        cutilSafeCall( cudaStreamSynchronize(stream_app) );
    else
        cutilSafeCall( cudaThreadSynchronize() );
	   
        cutilCheckError(cutStopTimer(hTimer));
        float gpuTime = cutGetTimerValue(hTimer) / (float)iterations;
    printf("Average GPU convolution time : %f msec //%f Mpixels/sec\n", gpuTime, 1e-6 * imageW * imageH / (gpuTime * 0.001));

    printf("Reading back GPU results...\n");
        cutilSafeCall( cudaMemcpyAsync(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, stream_app) );

    printf("Checking the results...\n");
        printf("...running convolutionRowCPU()\n");

        convolutionRowCPU(
            h_Buffer,
            h_Input,
            h_Kernel,
            imageW,
            imageH,
            KERNEL_RADIUS
        );

        printf("...running convolutionColumnCPU()\n");
        convolutionColumnCPU(
            h_OutputCPU,
            h_Buffer,
            h_Kernel,
            imageW,
            imageH,
            KERNEL_RADIUS
        );

        printf("...comparing the results\n");
        double sum = 0, delta = 0;
        for(unsigned i = 0; i < imageW * imageH; i++){
            delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
            sum   += h_OutputCPU[i] * h_OutputCPU[i];
        }
        double L2norm = sqrt(delta / sum);
        printf("Relative L2 norm: %E\n", L2norm);
    printf((L2norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");

    printf("Shutting down...\n");
        cutilSafeCall( cudaFree(d_Buffer ) );
        cutilSafeCall( cudaFree(d_Output) );
        cutilSafeCall( cudaFree(d_Input) );
        free(h_OutputGPU);
        free(h_OutputCPU);
        free(h_Buffer);
        free(h_Input);
        free(h_Kernel);

    cutilCheckError(cutDeleteTimer(hTimer));

    //cudaThreadExit();

    //cutilExit(argc, argv);
}
