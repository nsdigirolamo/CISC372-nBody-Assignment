#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

void compute () {

	int i, j, k;

	vector3* values = (vector3*) malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3** accels = (vector3**) malloc(sizeof(vector3*) * NUMENTITIES);

	for (i = 0; i < NUMENTITIES; i++) {
		accels[i] = &values[i * NUMENTITIES];
	}

	for (i = 0; i < NUMENTITIES; i++) {
		for (j = 0; j < NUMENTITIES; j++) {
			if (i == j) {

				FILL_VECTOR(accels[i][j], 0, 0, 0); 

			} else {

				vector3 distance;

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

	for (i = 0; i < NUMENTITIES; i++) {

		vector3 accel_sum={0,0,0};

		for (j = 0; j < NUMENTITIES; j++) {
			for (k = 0; k < 3; k++) {
				accel_sum[k] += accels[i][j][k];
			}
		}

		for (k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}

	free(accels);
	free(values);
}
