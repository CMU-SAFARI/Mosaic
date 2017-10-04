Requirements:

- NVIDIA CUDA SDK
- CUDA 4.0
- MAFIA benchmark Suite
-- Available at https://github.com/adwaitjog/mafia
- gcc 4.4 and g++ 4.8
-- Note that the Makefile in pthread_benchmark as well as NVIDIA CUDA SDK folder can potentially be updated to enforce gcc and g++ to use the same version. However, it is not supported at the moment. g++4.8 is needed for Mosaic main simulator as we have been using newer funcationalities of g++.

How to run Mosaic:

1) Modify CUDAHOME and NVIDIA_CUDA_SDK_LOCATION in v3.x/setup_environment to the location where NVIDIA CUDA SDK and CUDA 4.0 are installed.

2) Make the simulator by running make in the v3.x folder

3) To make the benchmarks, copy over files in Mosaic's v3.x/pthread_benchmark folder to the original pthread_benchmark. Then, make in the pthread_benchmark folder.
The updated pthread_benchmark allows unlimited number of concurrent applications

4) run ./gpgpu_ptx_sim [benchmark name list] 
4.1) [benchmark name list] should contain a list of benchmarks that will be run, seperate by space. For example, running ./gpgpu_ptx_sim HS HS CONS will run launch the simulation with three concurrently executing benchmarks, where there are two copies of HS (each with their own separate address space) and one copy of CONS across GPU SMs. Note that the current policy split the GPU cores evenly for each application.


Please send questions to rachata@cmu.edu

The current version of the simulator is provided as is, and should be treated as an alpha version. 

Please cite the following paper when using Mosaic simulator:

- Rachata Ausavarungnirun, Joshua Landgraf, Vance Miller, Saugata Ghose, Jayneel Gandhi, Christopher J. Rossbach, and Onur Mutlu. "Mosaic: A GPU Memory Manager with Application-Transparent Support for Multiple Page Sizes". In the Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO 2017), Boston, MA, October 2017. 

Please credit the following paper when using the Mafia benchmark suite:

- Adwait Jog, Onur Kayiran, Tuba Kesten, Ashutosh Pattnaik, Evgeny Bolotin, Niladrish Chatterjee, Stephen W. Keckler, Mahmut T. Kandemir, Chita R. Das. "Anatomy of GPU Memory System for Multi-Application Execution". In the Proceedings of 1st International Symposium on Memory Systems (MEMSYS), Washington, DC, Oct 2015

