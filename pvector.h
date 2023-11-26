#ifndef __TYPES_H__
#define __TYPES_H__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}

extern vector3* host_velocities;
extern vector3* host_positions;
extern double* host_masses;

extern vector3* device_velocities;
extern vector3* device_positions;
extern double* device_masses;

extern vector3** dists;
extern vector3** accels;
extern vector3* accel_sums;

#endif