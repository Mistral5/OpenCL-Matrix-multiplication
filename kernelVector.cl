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

	const unsigned char currLocalCol = get_local_id(0);
	const unsigned char currLocalRow = get_local_id(1);

	floatType currElResultMatrix = (floatType)(0.0f);

	__local floatType localFM[LSIZE][LSIZE / vecWidth];
	__local float localSM[LSIZE][LSIZE + 1];

	for(unsigned int i = 0; i < normalColRow; i++)
	{
		unsigned int currLSIZE = i * LSIZE;

		if(currRow < rowQuantity && (currLocalCol * vecWidth + currLSIZE + vecWidth - 1) < colFirstRowSecond)
		{
			localFM[currLocalRow][currLocalCol] = vload4(0, firstMatrix + (currRow * colFirstRowSecond + currLocalCol * vecWidth + currLSIZE));
		}
		else if(currRow < rowQuantity && (currLocalCol * vecWidth + currLSIZE) < colFirstRowSecond)
		{
			unsigned int k = colFirstRowSecond - (currLocalCol * vecWidth + currLSIZE);
			if(k == 1)
			{
				localFM[currLocalRow][currLocalCol].s0 = firstMatrix[currRow * colFirstRowSecond + currLocalCol * vecWidth + currLSIZE];
				localFM[currLocalRow][currLocalCol].s123 = (float3)(0.0f);
			}
			if(k == 2)
			{
				localFM[currLocalRow][currLocalCol].s01 = vload2(0, firstMatrix + (currRow * colFirstRowSecond + currLocalCol * vecWidth + currLSIZE));
				localFM[currLocalRow][currLocalCol].s23 = (float2)(0.0f);
			}
			if(k == 3)
			{
				localFM[currLocalRow][currLocalCol].s012 = vload3(0, firstMatrix + (currRow * colFirstRowSecond + currLocalCol * vecWidth + currLSIZE));
				localFM[currLocalRow][currLocalCol].s3 = 0.0f;
			}
		}
		else
		{
			localFM[currLocalRow][currLocalCol] = (floatType)(0.0f);
		}

		for(unsigned int m = 0; m < vecWidth; m++)
		{
			if(currLocalRow < colFirstRowSecond - currLSIZE && currLocalCol * vecWidth + m < colQuantity)
				localSM[currLocalRow][currLocalCol * vecWidth + m] = secondMatrix[(currCol * vecWidth + m) * colFirstRowSecond + currLocalRow + currLSIZE];
			else
				localSM[currLocalRow][currLocalCol * vecWidth + m] = 0;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		float currElemFM;

		for(unsigned int j = 0; j < LSIZE / vecWidth; j++)
		{
			floatType elemFM = localFM[currLocalRow][j];

			for(unsigned int m = 0; m < vecWidth; m++)
			{
				switch(m)
				{
					case 0: currElemFM = elemFM.s0; break;
					case 1: currElemFM = elemFM.s1; break;
					case 2: currElemFM = elemFM.s2; break;
					case 3: currElemFM = elemFM.s3; break;
				}

				currElResultMatrix.s0 += currElemFM * localSM[j * vecWidth + m][currLocalCol * vecWidth];
				currElResultMatrix.s1 += currElemFM * localSM[j * vecWidth + m][currLocalCol * vecWidth + 1];
				currElResultMatrix.s2 += currElemFM * localSM[j * vecWidth + m][currLocalCol * vecWidth + 2];
				currElResultMatrix.s3 += currElemFM * localSM[j * vecWidth + m][currLocalCol * vecWidth + 3];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	int k = colQuantity - (currCol * vecWidth);

	if(currRow < rowQuantity && (currCol * vecWidth + vecWidth - 1) < colQuantity)
	{
		vstore4(currElResultMatrix, 0, resultMatrix + (currRow * colQuantity + currCol * vecWidth));
	}
	else
	{
		int k = colQuantity - currCol * vecWidth;

		if(k == 1)
			resultMatrix[currRow * colQuantity + currCol * vecWidth] = currElResultMatrix.s0;

		if(k == 2)
			vstore2(currElResultMatrix.s01, 0, resultMatrix + (currRow * colQuantity + currCol * vecWidth));

		if(k == 3)
			vstore3(currElResultMatrix.s012, 0, resultMatrix + (currRow * colQuantity + currCol * vecWidth));
	}
}