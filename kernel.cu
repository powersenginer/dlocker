/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    const int TILE_WIDTH = TILE_SIZE;
    __shared__ float  M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float  N[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;  
    int by = blockIdx.y;  
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    
    int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	
	
	int M_Width = m;
	int N_Width = n;
	int P_Width = n;
	
	float Pvalue = 0;
	
	for (int p = 0; p < (k - 1) / TILE_WIDTH + 1; ++p) {
		if(Row < M_Width && p* TILE_WIDTH+tx < k) {
			M[ty][tx] = A[Row * k + p * TILE_WIDTH + tx];} 
		else {
			M[ty][tx] = 0.0;}
		if (p*TILE_WIDTH+ty < k && Col < N_Width) {
			N[ty][tx] = B[(p*TILE_WIDTH + ty) *N_Width + Col];} 
		else {
			N[ty][tx] = 0.0;}
			__syncthreads();
		if(Row < P_Width && Col < P_Width) {
			for (int i = 0; i < TILE_WIDTH; ++i) {
				Pvalue += M[ty][i]  * N[i][tx];}
		}
		__syncthreads();
	} /* end of outer for loop */
	if (Row < M_Width && Col < N_Width) {
		C[Row*P_Width + Col] = Pvalue;} /* end of kernel */

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

	dim3 DimGrid((n-1)/BLOCK_SIZE + 1, (m-1)/BLOCK_SIZE + 1, 1);
	dim3 DimBlock(BLOCK_SIZE ,  BLOCK_SIZE, 1);
	
    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
  
	mysgemm<<<DimGrid  ,  DimBlock>>>(m, n, k, A, B, C);

}


