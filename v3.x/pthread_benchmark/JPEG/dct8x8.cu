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

/**
**************************************************************************
* \file dct8x8.cu
* \brief Contains entry point, wrappers to host and device code and benchmark.
*
* This sample implements forward and inverse Discrete Cosine Transform to blocks
* of image pixels (of 8x8 size), as in JPEG standard. The typical work flow is as 
* follows:
* 1. Run CPU version (Host code) and measure execution time;
* 2. Run CUDA version (Device code) and measure execution time;
* 3. Output execution timings and calculate CUDA speedup.
*/

#include "../benchmark_common.h"

#include "Common.h"


/**
*  The number of DCT kernel calls
*/
#ifdef __DEVICE_EMULATION__
#define BENCHMARK_SIZE	1
#else
#define BENCHMARK_SIZE	10
#endif


/**
*  The PSNR values over this threshold indicate images equality
*/
#define PSNR_THRESHOLD_EQUAL	40


/**
*  Texture reference that is passed through this global variable into device code.
*  This is done because any conventional passing through argument list way results 
*  in compiler internal error. 2008.03.11
*/
texture<float, 2, cudaReadModeElementType> TexSrc;


// includes kernels
#include "dct8x8_kernel1.cu"
#include "dct8x8_kernel2.cu"
#include "dct8x8_kernel_short.cu"
#include "dct8x8_kernel_quantization.cu"


float encode(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size, cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag)
{
	//allocate host buffers for DCT and other data
  int StrideF;
  float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

	//convert source image to float representation
  CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
  AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

	//allocate device memory
  float *SrcDst;
  size_t DeviceStride;
  cutilSafeCall(cudaMallocPitch((void **)(&SrcDst), &DeviceStride, Size.width * sizeof(float), Size.height));
  DeviceStride /= sizeof(float);

	//copy from host memory to device
  cutilSafeCall(cudaMemcpy2D(SrcDst, DeviceStride * sizeof(float), 
                ImgF1, StrideF * sizeof(float), 
                                        Size.width * sizeof(float), Size.height,
                                            cudaMemcpyHostToDevice));

	//create and start CUDA timer
  unsigned int timerCUDA = 0;
  cutilCheckError(cutCreateTimer(&timerCUDA));
  cutilCheckError(cutResetTimer(timerCUDA));

	//setup execution parameters
  dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
  dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH/8, KER2_BLOCK_HEIGHT/8);

	//perform block-wise DCT processing and benchmarking
  cutilCheckError(cutStartTimer(timerCUDA));
  CUDAkernel2DCT<<< GridFullWarps, ThreadsFullWarps, 0, stream_app >>>(SrcDst, (int) DeviceStride);
  
  pthread_mutex_unlock (mutexapp);
  if(flag)
      cutilSafeCall( cudaStreamSynchronize(stream_app) );
  else
      cutilSafeCall( cudaThreadSynchronize() );
  //cutilSafeCall( cudaStreamSynchronize(stream_app) );
  //cutilSafeCall( cudaThreadSynchronize() );
  
  //cutilCheckError(cutStopTimer(timerCUDA));
  cutilCheckMsg("Kernel execution failed");

	// finalize CUDA timer
  //float TimerCUDASpan = cutGetAverageTimerValue(timerCUDA);
  //cutilCheckError(cutDeleteTimer(timerCUDA));
  pthread_mutex_lock (mutexapp);

	//setup execution parameters for quantization
  dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
  dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

	// execute Quantization kernel
  CUDAkernelQuantizationFloat<<< GridSmallBlocks, ThreadsSmallBlocks, 0, stream_app>>>(SrcDst, (int) DeviceStride);
  
  pthread_mutex_unlock (mutexapp);
  //cutilSafeCall( cudaStreamSynchronize(stream_app) );
  cutilCheckMsg("Kernel execution failed");
  //cutilSafeCall( cudaThreadSynchronize() );
  if(flag)
      cutilSafeCall( cudaStreamSynchronize(stream_app) );
  else
      cutilSafeCall( cudaThreadSynchronize() );
  //copy quantized image block to host
  cutilSafeCall(cudaMemcpy2D(ImgF1, StrideF * sizeof(float), 
                SrcDst, DeviceStride * sizeof(float), 
                                              Size.width * sizeof(float), Size.height,
                                                  cudaMemcpyDeviceToHost
                                                  ));
  cutilCheckError(cutStopTimer(timerCUDA));
  float TimerCUDASpan = cutGetAverageTimerValue(timerCUDA);
  cutilCheckError(cutDeleteTimer(timerCUDA));

	//convert image back to byte representation
  AddFloatPlane(128.0f, ImgF1, StrideF, Size);
  CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

	//clean up memory
  cutilSafeCall(cudaFree(SrcDst));
  FreePlane(ImgF1);

	//return time taken by the operation
  return TimerCUDASpan;
}

float decode(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size, cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag)
{
  	//allocate float buffers for DCT and other data
  int StrideF;
  float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
  float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

	//convert source image to float representation
  CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
  AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

	//perform block-wise DCT processing and benchmarking
  for (int i=0; i<BENCHMARK_SIZE; i++)
  {
    computeDCT8x8Gold1(ImgF1, ImgF2, StrideF, Size);
  }

	//perform quantization
  quantizeGoldFloat(ImgF2, StrideF, Size);

	//allocate device memory
  float *SrcDst;
  size_t DeviceStride;
  cutilSafeCall(cudaMallocPitch((void **)(&SrcDst), &DeviceStride, Size.width * sizeof(float), Size.height));
  DeviceStride /= sizeof(float);

	//copy from host memory to device
  cutilSafeCall(cudaMemcpy2D(SrcDst, DeviceStride * sizeof(float), 
                ImgF2, StrideF * sizeof(float), 
                                        Size.width * sizeof(float), Size.height,
                                            cudaMemcpyHostToDevice
                                            ));

	//create and start CUDA timer
  unsigned int timerCUDA = 0;
  cutilCheckError(cutCreateTimer(&timerCUDA));
  cutilCheckError(cutResetTimer(timerCUDA));

	//setup execution parameters
  dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
  dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH/8, KER2_BLOCK_HEIGHT/8);

	//perform block-wise IDCT processing
  cutilCheckError(cutStartTimer(timerCUDA));
  CUDAkernel2IDCT<<< GridFullWarps, ThreadsFullWarps, 0, stream_app >>>(SrcDst, (int) DeviceStride);
  
  pthread_mutex_unlock (mutexapp);
  //cutilSafeCall( cudaStreamSynchronize(stream_app) );
  //cutilSafeCall( cudaThreadSynchronize() );
  if(flag)
      cutilSafeCall( cudaStreamSynchronize(stream_app) );
  else
      cutilSafeCall( cudaThreadSynchronize() );
  
  cutilCheckMsg("Kernel execution failed");

	//copy quantized image block to host
  cutilSafeCall(cudaMemcpy2D(ImgF2, StrideF * sizeof(float), 
                SrcDst, DeviceStride * sizeof(float), 
                                              Size.width * sizeof(float), Size.height,
                                                  cudaMemcpyDeviceToHost
                                                  ));
  cutilCheckError(cutStopTimer(timerCUDA));
  float TimerCUDASpan = cutGetAverageTimerValue(timerCUDA);
  cutilCheckError(cutDeleteTimer(timerCUDA));

	//convert image back to byte representation
  AddFloatPlane(128.0f, ImgF2, StrideF, Size);
  CopyFloat2Byte(ImgF2, StrideF, ImgDst, Stride, Size);

	//clean up memory
  cutilSafeCall(cudaFree(SrcDst));
  FreePlane(ImgF1);
  FreePlane(ImgF2);

	//return time taken by the operation
  return TimerCUDASpan;
}

void usage() {
  printf("Usage:\n");
  printf("  jpeg [mode] --file=[input_file]\n");
  printf("\n");
  printf("  [mode]\n");
  printf("    --encode - performs dct and quantization\n");
  printf("    --decode - performs idct\n");
  printf("\n");
  printf("  [input_file] - name of the file to work with\n");
  printf("\n");
  exit(0);
}

/**
**************************************************************************
*  Program entry point
*
* \param argc		[IN] - Number of command-line arguments
* \param argv		[IN] - Array of command-line arguments
*  
* \return Status code
*/
int main_JPEG(cudaStream_t stream_app, pthread_mutex_t *mutexapp, bool flag)
//int main(int argc, char** argv)
{

	//initialize CUDA
    //if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		//cutilDeviceInit(argc, argv);
	//else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
  
  //check if we are encoding or decoding
    bool encoding;

  /*if(cutCheckCmdLineFlag(argc, (const char**)argv, "encode")) {
    //we are encoding
    encoding = true;
  } else if(cutCheckCmdLineFlag(argc, (const char**)argv, "decode")) {
    //we are decoding
    encoding = false;
    //get filename
  } else {
    //encode or decode were not specified
    usage();
  }
  
   //get filename
   char *SampleImageFname;
   if(!cutGetCmdLineArgumentstr(argc, (const char**)argv, "file", &SampleImageFname)) {
     usage();
   }
   if(strlen(SampleImageFname)==0) {
    usage();
   }*/

     
    encoding = false; // HARD CODING
	//preload image (acquire dimensions)
	int ImgWidth, ImgHeight;
	ROI ImgSize;
        int res = PreLoadBmp("JPEG/data/image.bmp", &ImgWidth, &ImgHeight);
	ImgSize.width = ImgWidth;
	ImgSize.height = ImgHeight;

	if (res)
	{
		printf("Error: Image file not found or invalid!\n");
		return 1;
	}

	//check image dimensions are multiples of BLOCK_SIZE
	if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
	{
		printf("Error: Input image dimensions must be multiples of 8!\n");
		return 1;
	}

	//allocate image buffers
	int ImgStride;
	byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
	byte *ImgDstCUDA1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

	//load sample image
        LoadBmpAsGray("JPEG/data/image.bmp", ImgStride, ImgSize, ImgSrc);

	//
	// RUNNING WRAPPERS
	//

  float time;
  if(encoding) {
	time = encode(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize, stream_app, mutexapp, flag);
  } else {
	time = decode(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize, stream_app, mutexapp, flag);
  }
  printf("%f\n",time);

	//
	// Finalization
	//

	//release byte planes
	FreePlane(ImgSrc);
	FreePlane(ImgDstCUDA1);

	//finalize
	//cudaThreadExit();

	//cutilExit(argc, argv);
	printf("SUCCESS\n");
	return 0;
}
