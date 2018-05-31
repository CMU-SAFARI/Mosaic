
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "../benchmark_common.h"

using namespace std;
// ****************************************************************************
// Function: triad
//
// Purpose:
//   A simple vector addition kernel
//   C = A + s*B
//
// Arguments:
//   A,B - input vectors
//   C - output vectors
//   s - scalar
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
__global__ void triad(float* A, float* B, float* C, float s) {
  int gid = threadIdx.x + (blockIdx.x * blockDim.x);
  C[gid] = A[gid] + s * B[gid];
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.

int main_TRD(cudaStream_t stream_app, pthread_mutex_t* mutexapp, bool flag) {
  const bool verbose = "";
  // const int n_passes = op.getOptionInt("passes");
  const int n_passes = 10;

  // 256k through 8M bytes
  const int nSizes = 9;
  const size_t blockSizes[] = {64,   128,  256,  512,  1024,
                               2048, 4096, 8192, 16384};
  const size_t memSize = 16384;
  const size_t numMaxFloats = 1024 * memSize / 4;
  const size_t halfNumFloats = numMaxFloats / 2;

  // Create some host memory pattern
  srand48(8650341L);
  float* h_mem;
  cudaMallocHost((void**)&h_mem, sizeof(float) * numMaxFloats);
  // CHECK_CUDA_ERROR();

  // Allocate some device memory
  float *d_memA0, *d_memB0, *d_memC0;
  cudaMalloc((void**)&d_memA0, blockSizes[nSizes - 1] * 1024);
  cudaMalloc((void**)&d_memB0, blockSizes[nSizes - 1] * 1024);
  cudaMalloc((void**)&d_memC0, blockSizes[nSizes - 1] * 1024);
  // CHECK_CUDA_ERROR();

  float *d_memA1, *d_memB1, *d_memC1;
  cudaMalloc((void**)&d_memA1, blockSizes[nSizes - 1] * 1024);
  cudaMalloc((void**)&d_memB1, blockSizes[nSizes - 1] * 1024);
  cudaMalloc((void**)&d_memC1, blockSizes[nSizes - 1] * 1024);
  // CHECK_CUDA_ERROR();

  float scalar = 1.75f;

  const size_t blockSize = 128;
  pthread_mutex_unlock(mutexapp);
  // Number of passes. Use a large number for stress testing.
  // A small value is sufficient for computing sustained performance.
  char sizeStr[256];
  for (int pass = 0; pass < n_passes; ++pass) {
    // Step through sizes forward
    for (int i = 0; i < nSizes; ++i) {
      int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
      for (int j = 0; j < halfNumFloats; ++j)
        h_mem[j] = h_mem[halfNumFloats + j] = (float)(drand48() * 10.0);

      // Copy input memory to the device
      if (verbose)
        cout << ">> Executing Triad with vectors of length " << numMaxFloats
             << " and block size of " << elemsInBlock << " elements."
             << "\n";
      sprintf(sizeStr, "Block:%05ldKB", blockSizes[i]);

      // start submitting blocks of data of size elemsInBlock
      // overlap the computation of one block with the data
      // download for the next block and the results upload for
      // the previous block
      int crtIdx = 0;
      size_t globalWorkSize = elemsInBlock / blockSize;

      // cudaStream_t streams[2];
      // cudaStreamCreate(&streams[0]);
      // cudaStreamCreate(&streams[1]);
      // CHECK_CUDA_ERROR();

      // int TH = Timer::Start();

      cudaMemcpyAsync(d_memA0, h_mem, blockSizes[i] * 1024,
                      cudaMemcpyHostToDevice, stream_app);
      cudaMemcpyAsync(d_memB0, h_mem, blockSizes[i] * 1024,
                      cudaMemcpyHostToDevice, stream_app);
      // CHECK_CUDA_ERROR();
      pthread_mutex_lock(mutexapp);
      triad<<<globalWorkSize, blockSize, 0, stream_app>>>(d_memA0, d_memB0,
                                                          d_memC0, scalar);
      pthread_mutex_unlock(mutexapp);
      if (flag)
        cutilSafeCall(cudaStreamSynchronize(stream_app));

      if (elemsInBlock < numMaxFloats) {
        // start downloading data for next block
        cudaMemcpyAsync(d_memA1, h_mem + elemsInBlock, blockSizes[i] * 1024,
                        cudaMemcpyHostToDevice, stream_app);
        cudaMemcpyAsync(d_memB1, h_mem + elemsInBlock, blockSizes[i] * 1024,
                        cudaMemcpyHostToDevice, stream_app);
        // CHECK_CUDA_ERROR();
      }

      int blockIdx = 1;
      unsigned int currStream = 1;
      while (crtIdx < numMaxFloats) {
        currStream = blockIdx & 1;
        // Start copying back the answer from the last kernel
        if (currStream) {
          cudaMemcpyAsync(h_mem + crtIdx, d_memC0, elemsInBlock * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_app);
        } else {
          cudaMemcpyAsync(h_mem + crtIdx, d_memC1, elemsInBlock * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_app);
        }

        crtIdx += elemsInBlock;

        if (crtIdx < numMaxFloats) {
          // Execute the kernel
          pthread_mutex_lock(mutexapp);
          if (currStream) {
            triad<<<globalWorkSize, blockSize, 0, stream_app>>>(
                d_memA1, d_memB1, d_memC1, scalar);
          } else {
            triad<<<globalWorkSize, blockSize, 0, stream_app>>>(
                d_memA0, d_memB0, d_memC0, scalar);
          }
          pthread_mutex_unlock(mutexapp);
        }

        if (flag)
          cutilSafeCall(cudaStreamSynchronize(stream_app));

        if (crtIdx + elemsInBlock < numMaxFloats) {
          // Download data for next block
          if (currStream) {
            cudaMemcpyAsync(d_memA0, h_mem + crtIdx + elemsInBlock,
                            blockSizes[i] * 1024, cudaMemcpyHostToDevice,
                            stream_app);
            cudaMemcpyAsync(d_memB0, h_mem + crtIdx + elemsInBlock,
                            blockSizes[i] * 1024, cudaMemcpyHostToDevice,
                            stream_app);
          } else {
            cudaMemcpyAsync(d_memA1, h_mem + crtIdx + elemsInBlock,
                            blockSizes[i] * 1024, cudaMemcpyHostToDevice,
                            stream_app);
            cudaMemcpyAsync(d_memB1, h_mem + crtIdx + elemsInBlock,
                            blockSizes[i] * 1024, cudaMemcpyHostToDevice,
                            stream_app);
          }
        }
        blockIdx += 1;
        currStream = !currStream;
      }
      if (flag)
        cutilSafeCall(cudaStreamSynchronize(stream_app));
      else
        cudaThreadSynchronize();
      // double time = Timer::Stop(TH, "thread synchronize");

      // double triad = ((double)numMaxFloats * 2.0) / (time*1e9);
      // resultDB.AddResult("TriadFlops", sizeStr, "GFLOP/s", triad);

      // double bdwth = ((double)numMaxFloats*sizeof(float)*3.0)
      //    / (time*1000.*1000.*1000.);
      // resultDB.AddResult("TriadBdwth", sizeStr, "GB/s", bdwth);

      // Checking memory for correctness. The two halves of the array
      // should have the same results.
      if (verbose)
        cout << ">> checking memory\n";
      for (int j = 0; j < halfNumFloats; ++j) {
        if (h_mem[j] != h_mem[j + halfNumFloats]) {
          cout << "Error; hostMem[" << j << "]=" << h_mem[j]
               << " is different from its twin element hostMem["
               << (j + halfNumFloats) << "]: " << h_mem[j + halfNumFloats]
               << "stopping check\n";
          break;
        }
      }
      if (verbose)
        cout << ">> finish!" << endl;

      // Zero out the test host memory
      for (int j = 0; j < numMaxFloats; ++j)
        h_mem[j] = 0.0f;
    }
  }

  // Cleanup
  cudaFree(d_memA0);
  cudaFree(d_memB0);
  cudaFree(d_memC0);
  cudaFree(d_memA1);
  cudaFree(d_memB1);
  cudaFree(d_memC1);
  cudaFreeHost(h_mem);
  return 0;
}
