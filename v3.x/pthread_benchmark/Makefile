TARGET := gpgpu-sim

# Cuda source files (compiled with nvcc)
CUFILES		:= LUH/lulesh.cu SPMV/main_spmv.cu TRD/Triad.cu SCAN/Scan.cu CFD/euler3d.cu \
    SC/streamcluster_cuda.cu QTC/QTC.cu RED/Reduction.cu RAY/rayTracing.cu LIB/libor.cu \
    CONS/convolutionSeparable.cu BP/backprop_cuda.cu SAD/main_sad.cu LUD/cuda/lud.cu \
    SRAD/main_srad.cu FFT/fft.cu LPS/laplace3d.cu FWT/fastWalshTransform.cu NN/NN.cu NW/needle.cu \
    SCP/scalarProd.cu JPEG/dct8x8.cu BFS2/bfs.cu GUPS/CudaRandomAccess.cu HISTO/histogram256.cu \
    3DS/threeDS.cu MM/mm.cu HS/hotspot.cu BLK/BlackScholes.cu
# MUM/mummergpu.cu MUM/common.cu

# C/C++ source files (compiled with gcc / c++)
CPPFILES		:= main.cpp SPMV/gpu_info.cpp SPMV/file.cpp SC/streamcluster_cuda_cpu.cpp \
    QTC/libdata.cpp RAY/EasyBMP.cpp RAY/makebmp.cpp CONS/main.cpp \
    CONS/convolutionSeparable_gold.cpp BP/backprop.cpp FWT/fastWalshTransform_gold.cpp \
    LPS/laplace3d_gold.cpp SCP/scalarProd_gold.cpp JPEG/BmpUtil.cpp JPEG/DCT8x8_Gold.cpp \
    MM/io.cpp BLK/BlackScholes_gold.cpp
#MUM/mummergpu_main.cpp MUM/mummergpu_gold.cpp MUM/suffix-tree.cpp MUM/PoolMalloc.cpp

CFILES		:= SAD/file.c SAD/image.c FFT/parboil.c LUD/common/common.c SRAD/graphics.c SRAD/timer.c

CC := gcc
CCFLAGS := -pthread -Wall -g

CPP := g++
CPPFLAGS :=-pthread -Wall -g -std=c++11

NVCC := nvcc
CUFLAGS := -po maxrregcount=16 -arch sm_20 -g

INC := -I../src/gpgpu-sim -I$(CUDAHOME)/include -I$(NVIDIA_CUDA_SDK_LOCATION)/C/common/inc

CULIB := -L$(NVIDIA_CUDA_SDK_LOCATION)/C/lib -lcutil_x86_64

################################################################################
INTERMEDIATE_FILES := *.c_o *.cu_o *.cpp_o *.i *.ii *.o *.fatbin* *.cudafe* *.cu.cpp *.cubin \
    *.hash *.ptx
# Objects
COBJ := $(patsubst %.c, %.c_o, $(CFILES))
CUOBJ := $(patsubst %.cu, %.cu_o, $(CUFILES))
CPPOBJ := $(patsubst %.cpp, %.cpp_o, $(CPPFILES))

GPGPU_SIM_LIB := -L$(GPGPUSIM_ROOT)/lib/$(GPGPUSIM_BUILD_TYPE) \
                 -L$(GPGPUSIM_ROOT)/build/$(GPGPUSIM_BUILD_TYPE) \
                 -L$(GPGPUSIM_ROOT)/build/$(GPGPUSIM_BUILD_TYPE)/gpgpu-sim \
                 -L$(GPGPUSIM_ROOT)/build/$(GPGPUSIM_BUILD_TYPE)/cuda-sim \
                 -lcudart -lgpgpu_ptx_sim -lgpgpusim -lgpu_uarch_sim

PTX_SIM_LIB_BIN := $(GPGPUSIM_ROOT)/build/$(GPGPUSIM_BUILD_TYPE)/cuda-sim/libgpgpu_ptx_sim.a
GPGPU_SIM_LIB_BIN := $(GPGPUSIM_ROOT)/build/$(GPGPUSIM_BUILD_TYPE)/libgpgpusim.a
GPGPU_ARCH_LIB_BIN := $(GPGPUSIM_ROOT)/build/$(GPGPUSIM_BUILD_TYPE)/gpgpu-sim/libgpu_uarch_sim.a
################################################################################
# Rules and targets

print-%  : ; @echo $* = $($*)

all: $(PTX_SIM_LIB_BIN) $(GPGPU_SIM_LIB_BIN) $(GPGPU_ARCH_LIB_BIN) $(CPPOBJ) $(CUOBJ) $(COBJ)
	$(CPP) $(CPPFLAGS) $(INC) $^ $(CULIB) $(GPGPU_SIM_LIB) -o $(TARGET)

$(COBJ): %.c_o: %.c
	$(CC) $(CCFLAGS) $(INC) $^ -c -o $@

$(CUOBJ): %.cu_o: %.cu
	$(NVCC) $(CUFLAGS) $(INC) $^ -c $(CULIB) -o $@

$(CPPOBJ): %.cpp_o: %.cpp
	$(CPP) $(CPPFLAGS) $(INC) $^ -c $(CULIB) -o $@

.PHONY: clean
clean:
	-rm -f $(INTERMEDIATE_FILES) $(COBJ) $(CUOBJ) $(CPPOBJ) $(TARGET)

