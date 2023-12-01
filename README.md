# University of Delaware CISC 372 nBody Problem

Nicholas DiGirolamo

A parallel and serial implementation of a solution to the nbody problem. The serial implentation was provided by Professor Silber. The parallel implementation is a heavily modified version of the serial implentation.

Here are some links that I found helpful in my parallel implementation:

- https://docs.nvidia.com/cuda/cuda-c-programming-guide
- https://docs.nvidia.com/cuda/floating-point
- https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf 

I tried to keep the serial code as close to the original as possible, but I have made a few minor additions to it: 
- The printSystemAlt() function on line 92 in nbody.c 
- A call to printSystemAlt() on lines 130 to 134 in nbody.c
- The out directory and its contents

Use the ```make``` command to compile and run either version of the program in its respective directory. I have also included a few scripts that make comparing outputs easier.


