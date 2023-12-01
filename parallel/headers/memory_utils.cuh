#include "vector.cuh"

#ifndef __MEMORY_UTILS_CUH__
#define __MEMORY_UTILS_CUH__

// Functions

void initHostMemory ();
void initDeviceMemory ();
void copyHostToDevice ();
void copyDeviceToHost ();
void freeHostMemory ();
void freeDeviceMemory ();  

// Host Memory

extern vector3* host_velocities;
extern vector3* host_positions;
extern double* host_masses;

// Device Memory

extern vector3* device_velocities;
extern vector3* device_positions;
extern double* device_masses;

extern vector3** accels;

#endif