#ifndef __PCOMPUTE_H__
#define __PCOMPUTE_H__

void compute ();
void initDeviceMemory (int numObjects);
void freeDeviceMemory ();
void transferMemoryFromDeviceToHost (int numObjects);

#endif