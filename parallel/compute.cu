#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

extern __shared__ vector3 dists[];

/**
 * TODO: Look into how if statements effect warps. I don't think it matters too much for
 * simple if statements like below, but I could offer an extremely small amount of
 * speed.
 */

__global__ void calcAccels (vector3** accels, vector3* positions, double* masses) {

	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	int offset = blockDim.x * blockIdx.y;
	int col = offset + thread_id;
	int axis = threadIdx.y;

	if (NUMENTITIES <= col) return;

	if (row == col) {

		accels[row][col][axis] = 0;

	} else {

		dists[thread_id][axis] = positions[row][axis] - positions[col][axis];

		/**
		printf("row %d thread_id %d col %d axis %d dists[thread_id][axis] %30.5lf = positions[row][axis] %30.5lf - positions[col][axis] %30.5lf\n",
			row,
			thread_id,
			col,
			axis,
			dists[thread_id][axis],
			positions[row][axis],
			positions[col][axis]
		);
		*/

		__syncthreads();

		#ifdef STRICT_MATH

		// This reduces the number of floating point differences and makes the math match the serial version more closely.
		// See https://docs.nvidia.com/cuda/floating-point for more information.

		double magnitude_sq = __dadd_rn(__dadd_rn(__dmul_rn(dists[thread_id][0], dists[thread_id][0]), __dmul_rn(dists[thread_id][1], dists[thread_id][1])), __dmul_rn(dists[thread_id][2], dists[thread_id][2]));

		#else

		double magnitude_sq = dists[thread_id][0] * dists[thread_id][0] + dists[thread_id][1] * dists[thread_id][1] + dists[thread_id][2] * dists[thread_id][2];

		#endif

		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[col] / magnitude_sq;
		accels[row][col][axis] = accelmag * dists[thread_id][axis] / magnitude;

		#ifdef ACCEL_DEBUG
		printf("position[%d] {%2.10e, %2.10e, %2.10e} position[%d] {%2.10e, %2.10e, %2.10e}\n", row, positions[row][0], positions[row][1], positions[row][2], col, positions[col][0], positions[col][1], positions[col][2]);
		printf("distance {%2.10e, %2.10e, %2.10e}\n", dists[thread_id][0], dists[thread_id][1], dists[thread_id][2]);
		printf("magnitude_sq %2.10e magnitude %2.10e accelmag %2.10e\n", magnitude_sq, magnitude, accelmag);
		printf("accels[%d][%d]={%2.10e, %2.10e, %2.10e}\n\n", row, col, accels[row][col][0], accels[row][col][1], accels[row][col][2]);
		#endif

	}
}

/**
 * TODO: Make sumAccels() work faster. Right now a single thread is summing up
 * an entire row. Last time I tried addressing this the floating point issues got
 * in my way. Maybe try revisiting this?
 */

__global__ void sumAccels (vector3** accels, vector3* accel_sums) {

	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	int offset = blockDim.x * blockIdx.y;
	int col = offset + thread_id;
	int axis = threadIdx.y;

	if (col != 0) return;

	double accel_sum = 0;

	for (int i = 0; i < NUMENTITIES; i++) {
		accel_sum += accels[row][i][axis];
	}

	accel_sums[row][axis] = accel_sum;
}

__global__ void calcChanges (vector3* accel_sums, vector3* velocities, vector3* positions) {

	int row = blockIdx.x;
	int thread_id = threadIdx.x;
	int offset = blockDim.x * blockIdx.y;
	int col = offset + thread_id;
	int axis = threadIdx.y;

	if (col != 0) return;

	#ifdef STRICT_MATH

	velocities[row][axis] = __dadd_rn(__dmul_rn(accel_sums[row][axis], INTERVAL), velocities[row][axis]);
	positions[row][axis] = __dadd_rn(__dmul_rn(velocities[row][axis], INTERVAL), positions[row][axis]);

	#else

	velocities[row][axis] += accel_sums[row][axis] * INTERVAL;
	positions[row][axis] += velocities[row][axis] * INTERVAL;

	#endif
}

void compute () {

	dim3 blocks(NUMENTITIES, blocks_per_row);
	dim3 threads(threads_per_block / WARP_GROUP_SIZE, WARP_GROUP_SIZE);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error before compute! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	#endif

	// Calculate Accelerations
	calcAccels<<<blocks, threads, (threads_per_block / WARP_GROUP_SIZE) * sizeof(vector3)>>>(accels, device_positions, device_masses);
	#ifdef DEBUG
	cudaError_t calc_accels_error = cudaGetLastError();
	if (calc_accels_error != cudaSuccess) 
		printf("calcAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_accels_error),
			cudaGetErrorString(calc_accels_error)
		);
	#endif
	cudaDeviceSynchronize();

	// Sum Accelerations
	sumAccels<<<blocks, threads>>>(accels, accel_sums);
	#ifdef DEBUG
	cudaError_t sum_accels_error = cudaGetLastError();
	if (sum_accels_error != cudaSuccess) 
		printf("sumAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(sum_accels_error),
			cudaGetErrorString(sum_accels_error)
		);
	#endif
	cudaDeviceSynchronize();

	// Calculating Changes
	calcChanges<<<blocks, threads>>>(accel_sums, device_velocities, device_positions);
	#ifdef DEBUG
	cudaError_t calc_changes_error = cudaGetLastError();
	if (calc_changes_error != cudaSuccess) 
		printf("calcChanges kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_changes_error),
			cudaGetErrorString(calc_changes_error)
		);
	#endif
	cudaDeviceSynchronize();
}
