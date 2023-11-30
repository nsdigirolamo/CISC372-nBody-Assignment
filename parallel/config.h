
#ifndef __CONFIG_H__
#define __CONFIG_H__
#define NUMPLANETS      8
#define MINUTE 			60
#define HOUR 			MINUTE*60
#define DAY 			HOUR*24
#define WEEK 			DAY*7
#define YEAR 			DAY*365
// Configurables
#define NUMASTEROIDS 100
#define GRAV_CONSTANT 6.67e-11
#define MAX_DISTANCE 5000.0
#define MAX_VELOCITY 50000.0
#define MAX_MASS 938e18 // Approximate mass of Ceres
#define DURATION (10*YEAR)
#define INTERVAL DAY
// End Configurables

// My Constants
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

#define MINIMUM_DIM_SIZE 1
#define CALC_ACCELS_BLOCK_WIDTH 16
#define SPATIAL_DIMS 3
#define SUM_LENGTH 256

#define NUMENTITIES (NUMPLANETS+NUMASTEROIDS+1)
#endif