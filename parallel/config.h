
#ifndef __CONFIG_H__
#define __CONFIG_H__
#define NUMPLANETS      8
#define MINUTE 			60
#define HOUR 			MINUTE*60
#define DAY 			HOUR*24
#define WEEK 			DAY*7
#define YEAR 			DAY*365
// Configurables
#define NUMASTEROIDS 10
#define GRAV_CONSTANT 6.67e-11
#define MAX_DISTANCE 5000.0
#define MAX_VELOCITY 50000.0
#define MAX_MASS 938e18  // Approximate mass of Ceres
#define DURATION (10*YEAR)
#define INTERVAL DAY
// My Configurables
#define MAX_THREADS_PER_BLOCK 960 // This can be as high as 1024, but it should be divisible by WARP_GROUP_SIZE and THREADS_PER_WARP
#define THREADS_PER_WARP 32
#define MAX_WARPS_PER_BLOCK 30 // This can be as high as 32, but it should be divisible by WARP_GROUP_SIZE
#define WARP_GROUP_SIZE 3 
#define THREADS_PER_WARP_GROUP 96 // This should be WARP_GROUP_SIZE * THREADS_PER_WARP
#define MAX_WARP_GROUPS_PER_BLOCK 10 // This should be MAX_WARPS_PER_BLOCK / WARP_GROUP_SIZE
// End Configurables

#define NUMENTITIES (NUMPLANETS+NUMASTEROIDS+1)
#endif