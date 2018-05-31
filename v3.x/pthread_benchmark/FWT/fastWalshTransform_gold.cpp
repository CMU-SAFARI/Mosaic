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

///////////////////////////////////////////////////////////////////////////////
// CPU Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
extern "C" void fwtCPU(float* h_Output, float* h_Input, int log2N) {
  const int N = 1 << log2N;

  for (int pos = 0; pos < N; pos++)
    h_Output[pos] = h_Input[pos];

  // Cycle through stages with different butterfly strides
  for (int stride = N / 2; stride >= 1; stride >>= 1) {
    // Cycle through subvectors of (2 * stride) elements
    for (int base = 0; base < N; base += 2 * stride)
      // Butterfly index within subvector of (2 * stride) size
      for (int j = 0; j < stride; j++) {
        int i0 = base + j + 0;
        int i1 = base + j + stride;

        float T1 = h_Output[i0];
        float T2 = h_Output[i1];
        h_Output[i0] = T1 + T2;
        h_Output[i1] = T1 - T2;
      }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Straightforward Walsh Transform: used to test both CPU and GPU FWT
// Slow. Uses doubles because of straightforward accumulation
///////////////////////////////////////////////////////////////////////////////
extern "C" void slowWTcpu(float* h_Output, float* h_Input, int log2N) {
  const int N = 1 << log2N;

  for (int i = 0; i < N; i++) {
    double sum = 0;

    for (int j = 0; j < N; j++) {
      // Walsh-Hadamar quotent
      double q = 1.0;
      for (int t = i & j; t != 0; t >>= 1)
        if (t & 1)
          q = -q;

      sum += q * h_Input[j];
    }

    h_Output[i] = (float)sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Reference CPU dyadic convolution.
// Extremly slow because of non-linear memory access patterns (cache thrashing)
////////////////////////////////////////////////////////////////////////////////
extern "C" void dyadicConvolutionCPU(float* h_Result,
                                     float* h_Data,
                                     float* h_Kernel,
                                     int log2dataN,
                                     int log2kernelN) {
  const int dataN = 1 << log2dataN;
  const int kernelN = 1 << log2kernelN;

  for (int i = 0; i < dataN; i++) {
    double sum = 0;

    for (int j = 0; j < kernelN; j++)
      sum += h_Data[i ^ j] * h_Kernel[j];

    h_Result[i] = (float)sum;
  }
}
