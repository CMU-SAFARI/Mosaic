/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation and
 * any modifications thereto.  Any use, reproduction, disclosure, or
 * distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

/*
 * Walsh transforms belong to a class of generalized Fourier transformations.
 * They have applications in various fields of electrical engineering
 * and numeric theory. In this sample we demonstrate efficient implementation
 * of naturally-ordered Walsh transform
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */

#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../benchmark_common.h"

////////////////////////////////////////////////////////////////////////////////
// Reference CPU FWT
////////////////////////////////////////////////////////////////////////////////
extern "C" void fwtCPU(float* h_Output, float* h_Input, int log2N);
extern "C" void slowWTcpu(float* h_Output, float* h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(float* h_Result,
                                     float* h_Data,
                                     float* h_Kernel,
                                     int log2dataN,
                                     int log2kernelN);

////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "fastWalshTransform_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int log2Kernel = 7;

#ifndef __DEVICE_EMULATION__
const int log2Data = 23;
#else
const int log2Data = 15;
#endif
const int dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int DATA_SIZE = dataN * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(float);

const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
// int main(int argc, char *argv[]){
int main_FWT(cudaStream_t stream_app, pthread_mutex_t* mutexapp, bool flag) {
  float *h_Data, *h_Kernel, *h_ResultCPU, *h_ResultGPU;

  float *d_Data, *d_Kernel;

  double delta, ref, sum_delta2, sum_ref2, L2norm, gpuTime;

  unsigned int hTimer;
  int i;

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  // if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
  // cutilDeviceInit(argc, argv);
  // else
  cudaSetDevice(cutGetMaxGflopsDeviceId());

  cutilCheckError(cutCreateTimer(&hTimer));

  printf("Initializing data...\n");
  printf("...allocating CPU memory\n");
  cutilSafeMalloc(h_Kernel = (float*)malloc(KERNEL_SIZE));
  cutilSafeMalloc(h_Data = (float*)malloc(DATA_SIZE));
  cutilSafeMalloc(h_ResultCPU = (float*)malloc(DATA_SIZE));
  cutilSafeMalloc(h_ResultGPU = (float*)malloc(DATA_SIZE));
  printf("...allocating GPU memory\n");
  cutilSafeCall(cudaMalloc((void**)&d_Kernel, DATA_SIZE));
  cutilSafeCall(cudaMalloc((void**)&d_Data, DATA_SIZE));

  printf("...generating data\n");
  printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
  srand(2007);
  for (i = 0; i < kernelN; i++)
    h_Kernel[i] = (float)rand() / (float)RAND_MAX;

  for (i = 0; i < dataN; i++)
    h_Data[i] = (float)rand() / (float)RAND_MAX;

  cutilSafeCall(cudaMemset(d_Kernel, 0, DATA_SIZE));
  cutilSafeCall(cudaMemcpyAsync(d_Kernel, h_Kernel, KERNEL_SIZE,
                                cudaMemcpyHostToDevice, stream_app));
  cutilSafeCall(cudaMemcpyAsync(d_Data, h_Data, DATA_SIZE,
                                cudaMemcpyHostToDevice, stream_app));

  printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
  // cutilSafeCall( cudaThreadSynchronize() );
  // cutilSafeCall( cudaStreamSynchronize(stream_app) );
  if (flag)
    cutilSafeCall(cudaStreamSynchronize(stream_app));
  else
    cutilSafeCall(cudaThreadSynchronize());
  cutilCheckError(cutResetTimer(hTimer));
  cutilCheckError(cutStartTimer(hTimer));
  fwtBatchGPU(d_Data, 1, log2Data, stream_app);
  fwtBatchGPU(d_Kernel, 1, log2Data, stream_app);
  modulateGPU(d_Data, d_Kernel, dataN, stream_app);
  fwtBatchGPU(d_Data, 1, log2Data, stream_app);
  pthread_mutex_unlock(mutexapp);
  if (flag)
    cutilSafeCall(cudaStreamSynchronize(stream_app));
  else
    cutilSafeCall(cudaThreadSynchronize());
  // cutilSafeCall( cudaStreamSynchronize(stream_app) );
  // cutilSafeCall( cudaThreadSynchronize() );
  cutilCheckError(cutStopTimer(hTimer));
  gpuTime = cutGetTimerValue(hTimer);
  printf("GPU time: %f ms; GOP/s: %f\n", gpuTime,
         NOPS / (gpuTime * 0.001 * 1E+9));

  printf("Reading back GPU results...\n");
  cutilSafeCall(cudaMemcpyAsync(h_ResultGPU, d_Data, DATA_SIZE,
                                cudaMemcpyDeviceToHost, stream_app));

  if (flag)
    cutilSafeCall(cudaStreamSynchronize(stream_app));

  printf("Running straightforward CPU dyadic convolution...\n");
  dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

  printf("Comparing the results...\n");
  sum_delta2 = 0;
  sum_ref2 = 0;
  for (i = 0; i < dataN; i++) {
    delta = h_ResultCPU[i] - h_ResultGPU[i];
    ref = h_ResultCPU[i];
    sum_delta2 += delta * delta;
    sum_ref2 += ref * ref;
  }
  L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("L2 norm: %E\n", L2norm);
  printf((L2norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");

  printf("Shutting down...\n");
  cutilCheckError(cutDeleteTimer(hTimer));
  cutilSafeCall(cudaFree(d_Data));
  cutilSafeCall(cudaFree(d_Kernel));
  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);

  return 0;

  // cudaThreadExit();

  // cutilExit(argc, argv);
}
