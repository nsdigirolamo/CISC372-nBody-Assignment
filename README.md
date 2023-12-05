# University of Delaware CISC 372 nBody Problem

Nicholas DiGirolamo

A parallel and serial implementation of a solution to the nbody problem. The serial implentation was provided by Professor Silber. The parallel implementation is a heavily modified version of the serial implentation.

Here are some resources I found useful over the course of my development of the parallel version:

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)
- [Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/floating-point)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc)
- [A helpful stackoverflow post regarding cudaMallocPitch and cudaMemcpy2D](https://stackoverflow.com/a/43685290)

And here are some other misc links that I know I'm going to forget about if I don't include here:

- [Integer Division Rounding Up](https://stackoverflow.com/a/2422722)

I tried to keep the serial code as close to the original as possible, but I have made a few minor additions to it: 
- The printSystemAlt() function on line 92 in nbody.c 
- A call to printSystemAlt() on lines 130 to 134 in nbody.c
- A new compiler flag to enable or disable printSystemAlt() in the makefile

Use the ```make``` command to compile and run either version of the program in its respective directory. I have also included a few scripts that make comparing outputs easier.


