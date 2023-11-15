#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute () {

	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i, j, k;

	// defines space for 2D vector of vector3 pointers to hold acceleration effects
	vector3* values = (vector3*) malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	// Array of pointers. These pointers will point to sections in the above array of size NUMENTITIES that correspond to their respective entity.
	// For example, accels[1] will point to the first NUMENTITIES of values, and accels[6] will point to the 6th block of NUMENTITIES in values.
	vector3** accels = (vector3**) malloc(sizeof(vector3*) * NUMENTITIES);

	for (i = 0; i < NUMENTITIES; i++) {
		accels[i] = &values[i * NUMENTITIES];
	}

	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i = 0; i < NUMENTITIES; i++) {
		for (j = 0; j < NUMENTITIES; j++) {
			if (i == j) {
				// If the vector is itself, it has no acceleration effect
				FILL_VECTOR(accels[i][j], 0, 0, 0); 

			} else {

				vector3 distance;

				// Getting x, y, z distances between objects i and j
				for (k = 0; k < 3; k++) {
					distance[k] = hPos[i][k] - hPos[j][k];
				}

				
				double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
				double magnitude = sqrt(magnitude_sq);
				double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;

				FILL_VECTOR(
					accels[i][j],
					accelmag * distance[0] / magnitude,
					accelmag * distance[1] / magnitude,
					accelmag * distance[2] / magnitude
				);
			}
		}
	}

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i = 0; i < NUMENTITIES; i++) {

		vector3 accel_sum={0,0,0};

		for (j = 0; j < NUMENTITIES; j++) {
			for (k = 0; k < 3; k++) {
				accel_sum[k] += accels[i][j][k];
			}
		}

		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}

	free(accels);
	free(values);
}
