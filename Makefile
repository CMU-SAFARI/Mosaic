################################################################################
#
# Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= mergedapps
# Cuda source files (compiled with cudacc)
# MUM/mummergpu.cu MUM/common.cu LUH/lulesh.cu 
CUFILES		:= main.cu SPMV/main_spmv.cu TRD/Triad.cu SCAN/Scan.cu CFD/euler3d.cu SC/streamcluster_cuda.cu QTC/QTC.cu RED/Reduction.cu RAY/rayTracing.cu LIB/libor.cu CONS/convolutionSeparable.cu BP/backprop_cuda.cu SAD/main_sad.cu LUD/cuda/lud.cu SRAD/main_srad.cu FFT/fft.cu LPS/laplace3d.cu FWT/fastWalshTransform.cu NN/NN.cu NW/needle.cu SCP/scalarProd.cu JPEG/dct8x8.cu BFS2/bfs.cu GUPS/CudaRandomAccess.cu HISTO/histogram256.cu 3DS/threeDS.cu MM/mm.cu HS/hotspot.cu BLK/BlackScholes.cu
# CUDA dependency files

CU_DEPS		:= SC/streamcluster_header.cu CONS/convolutionSeparable_common.h BP/backprop_cuda_kernel.cu SAD/largerBlocks.cu SAD/sad4.cu LUD/cuda/lud_kernel.cu SRAD/extract_kernel.cu SRAD/prepare_kernel.cu SRAD/reduce_kernel.cu SRAD/compress_kernel.cu SRAD/srad_kernel.cu SRAD/srad2_kernel.cu FWT/fastWalshTransform_kernel.cu LPS/laplace3d_kernel.cu NW/needle_kernel.cu SCP/scalarProd_kernel.cu JPEG/dct8x8_kernel1.cu JPEG/dct8x8_kernel2.cu JPEG/dct8x8_kernel_short.cu JPEG/dct8x8_kernel_quantization.cu MM/sgemm_kernel.cu BLK/BlackScholes_kernel.cuh
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= SPMV/gpu_info.cpp SPMV/file.cpp SC/streamcluster_cuda_cpu.cpp QTC/libdata.cpp RAY/EasyBMP.cpp RAY/makebmp.cpp CONS/main.cpp CONS/convolutionSeparable_gold.cpp BP/backprop.cpp FWT/fastWalshTransform_gold.cpp LPS/laplace3d_gold.cpp SCP/scalarProd_gold.cpp JPEG/BmpUtil.cpp JPEG/DCT8x8_Gold.cpp MM/io.cpp BLK/BlackScholes_gold.cpp

#MUM/mummergpu_main.cpp MUM/mummergpu_gold.cpp MUM/suffix-tree.cpp MUM/PoolMalloc.cpp

CFILES		:= SAD/file.c SAD/image.c FFT/parboil.c LUD/common/common.c SRAD/graphics.c SRAD/timer.c
# Need good occupancy
CUDACCFLAGS     := -po maxrregcount=16




################################################################################
# Rules and targets
include ../../common/common_pthread.mk
  
