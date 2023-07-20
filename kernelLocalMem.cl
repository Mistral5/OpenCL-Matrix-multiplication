__kernel void matrixMultiplication(
									__global const float* firstMatrix,
									__global const float* secondMatrix,
									__global float* resultMatrix,
									const unsigned int colFirstRowSecond,
									const unsigned int colQuantity,
									const unsigned int rowQuantity,
									const unsigned int normalColRow)
{
	const unsigned int currCol = get_global_id(0);
	const unsigned int currRow = get_global_id(1);

	__local float localFM[LSIZE][LSIZE];
	__local float localSM[LSIZE][LSIZE + 1];

	const unsigned char currLocalCol = get_local_id(0);
	const unsigned char currLocalRow = get_local_id(1);

	float currElResultMatrix = 0.0f;

	for(unsigned int i = 0; i < normalColRow; i++)
	{
		unsigned int currLSIZE = i * LSIZE;

		if(currLocalCol < colFirstRowSecond - currLSIZE && currRow < rowQuantity)
			localFM[currLocalRow][currLocalCol] = firstMatrix[currRow * colFirstRowSecond + currLocalCol + currLSIZE];
		else
			localFM[currLocalRow][currLocalCol] = 0;
		
		if(currLocalRow < colFirstRowSecond - currLSIZE && currCol < colQuantity)
			localSM[currLocalCol][currLocalRow] = secondMatrix[currCol * colFirstRowSecond + currLocalRow + currLSIZE];
		else
			localSM[currLocalCol][currLocalRow] = 0;

		barrier(CLK_LOCAL_MEM_FENCE);

		for(unsigned int j = 0; j < LSIZE; j++)
			currElResultMatrix += localFM[currLocalRow][j] * localSM[currLocalCol][j];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(currRow < rowQuantity && currCol < colQuantity)
		resultMatrix[currRow * colQuantity + currCol] = currElResultMatrix;
}