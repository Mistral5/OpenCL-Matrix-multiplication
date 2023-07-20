__kernel void matrixMultiplication(
									__global const float* firstMatrix,
									__global const float* secondMatrix,
									__global float* resultMatrix,
									const unsigned int colFirstRowSecond,
									const unsigned int colQuantity)
{
	const unsigned int currCol = get_global_id(0);
	const unsigned int currRow = get_global_id(1);

	float currElResultMatrix = 0.0f;
	
	for(unsigned int i = 0; i < colFirstRowSecond; i++)
		currElResultMatrix += firstMatrix[currRow * colFirstRowSecond + i] * secondMatrix[currCol * colFirstRowSecond + i];
	
	resultMatrix[currRow * colQuantity + currCol] = currElResultMatrix;
}