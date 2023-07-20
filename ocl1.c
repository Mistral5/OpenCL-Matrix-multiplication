#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#pragma comment(lib, "opencl.lib")
#endif

#define CL_TARGET_OPENCL_VERSION 120

struct sizes
{
	unsigned int rowFirstMatrix;
	unsigned int colSecondMatrix;
	unsigned int colFirstRowSecond;
	unsigned int firstMatrix;
	unsigned int secondMatrix;
	unsigned int resultMatrix;
};

struct deviceInfo
{
	cl_device_id ID;
	cl_device_type type;
	cl_bool hostUnifiedMem;
	cl_uint sortID;
};

void errCodeOutput(cl_int errCode, char* errLog)
{
	fprintf(stderr, "Error code %d. The method that caused this is '%s'.\n", errCode, errLog);
}

unsigned char matrixSizing(FILE* inputFile, struct sizes* size)
{
	if (fscanf(inputFile, "%u ", &size->colSecondMatrix) < 1)
		return 1;

	if (fscanf(inputFile, "%u ", &size->colFirstRowSecond) < 1)
		return 1;

	if (fscanf(inputFile, "%u\n", &size->rowFirstMatrix) < 1)
		return 1;

	size->firstMatrix = size->rowFirstMatrix * size->colFirstRowSecond;
	size->secondMatrix = size->colFirstRowSecond * size->colSecondMatrix;
	size->resultMatrix = size->rowFirstMatrix * size->colSecondMatrix;

	return 0;
}

unsigned char readFile(FILE* inputFile, float* firstMatrix, float* secondMatrix, struct sizes* size)
{
	for (unsigned int i = 0; i < size->firstMatrix; i++)
	{
		if (fscanf(inputFile, "%f", &firstMatrix[i]) <= 0)
			return 1;
	}

	for (unsigned int i = 0; i < size->colFirstRowSecond; i++)
	{
		for (unsigned int j = 0; j < size->colSecondMatrix; j++)
		{
			if (fscanf(inputFile, "%f", &secondMatrix[j * size->colFirstRowSecond + i]) <= 0)
				return 1;
		}
	}

	return 0;
}

unsigned char writeFile(FILE* outputFile, float* matrix, struct sizes* size)
{
	if (fprintf(outputFile, "%u %u\n", size->colSecondMatrix, size->rowFirstMatrix) < 0)
		return 1;

	for (unsigned int i = 0; i < size->rowFirstMatrix; i++)
	{
		unsigned int currLine = size->colSecondMatrix * i;

		for (unsigned int j = 0; j < size->colSecondMatrix; j++)
		{
			unsigned int currElIndex = currLine + j;

			if (fprintf(outputFile, "%f ", matrix[currElIndex]) < 0)
				return 1;
		}

		if (fprintf(outputFile, "\n") < 0)
			return 1;
	}

	return 0;
}

unsigned char getDeviceNameAndMaxLocalGroupSize(cl_device_id device, size_t* maxLocalGroupSize, const int implementationType)
{
	cl_int errCodeReturn = CL_SUCCESS;
	size_t deviceNameSize = 0;

	errCodeReturn = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
	if (errCodeReturn != CL_SUCCESS)
	{
		errCodeOutput(errCodeReturn, "clGetDeviceInfo");
		return 1;
	}

	char* deviceName = (char*)malloc(sizeof(char) * deviceNameSize);
	if (deviceName == NULL)
	{
		fprintf(stderr, "Insufficient memory available!\n");
		free(deviceName);
		return 1;
	}

	errCodeReturn = clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName, 0);
	if (errCodeReturn != CL_SUCCESS)
	{
		errCodeOutput(errCodeReturn, "clGetDeviceInfo");
		free(deviceName);
		return 1;
	}

	if (implementationType != 1)
	{
		errCodeReturn = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxLocalGroupSize), maxLocalGroupSize, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clGetDeviceInfo");
			free(deviceName);
			return 1;
		}

		*maxLocalGroupSize = sqrt(*maxLocalGroupSize);

		if (*maxLocalGroupSize > 32 && implementationType == 2)
			*maxLocalGroupSize = 32;
	}

	printf("Device: %s\n", deviceName);

	free(deviceName);
	return 0;
}

unsigned char kernelFileProcessing(const char* kernelFilePath, size_t* kernelFileSize, unsigned char** kernelFileText)
{
	FILE* kernelFile = fopen(kernelFilePath, "rb");
	if (kernelFile == NULL)
		return 1;

	fseek(kernelFile, 0L, SEEK_END);
	*kernelFileSize = ftell(kernelFile);
	rewind(kernelFile);

	*kernelFileText = (unsigned char*)malloc(sizeof(unsigned char) * *kernelFileSize);

	if (fread(*kernelFileText, sizeof(char), *kernelFileSize, kernelFile) != *kernelFileSize)
		return 1;

	fclose(kernelFile);
	return 0;
}

cl_uint getDeviceNumber(cl_uint* platformNum)
{
	cl_int errCodeReturn = CL_SUCCESS;
	cl_uint deviceNum = 0;

	errCodeReturn = clGetPlatformIDs(0, NULL, platformNum);
	if (errCodeReturn != CL_SUCCESS) { errCodeOutput(errCodeReturn, "clGetPlatformIDs"); return 0; }

	if (!*platformNum)
		return 0;

	cl_platform_id* platformIDs = (cl_platform_id*)malloc(sizeof(cl_platform_id) * *platformNum);
	if (platformIDs == NULL)
	{
		fprintf(stderr, "Insufficient memory available!\n");
		free(platformIDs);
		return 0;
	}

	errCodeReturn = clGetPlatformIDs(*platformNum, platformIDs, NULL);
	if (errCodeReturn != CL_SUCCESS)
	{
		errCodeOutput(errCodeReturn, "clGetPlatformIDs");
		free(platformIDs);
		return 0;
	}

	for (cl_uint i = 0; i < *platformNum; i++)
	{
		cl_uint platformDeviceNum = 0;
		errCodeReturn = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &platformDeviceNum);
		//if (errCodeReturn != CL_SUCCESS && errCodeReturn != -1)
		//{
		//	errCodeOutput(errCodeReturn, "clGetDeviceIDs");
		//	free(platformIDs);
		//	return 0;
		//}

		deviceNum += platformDeviceNum;
	}

	free(platformIDs);
	return deviceNum;
}

unsigned char getDeviceInfo(struct deviceInfo* devices, cl_uint deviceNum, cl_uint platformNum)
{
	cl_int errCodeReturn = CL_SUCCESS;

	cl_platform_id* platformIDs = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformNum);
	if (platformIDs == NULL)
	{
		fprintf(stderr, "Insufficient memory available!\n");
		free(platformIDs);
		return 0;
	}

	errCodeReturn = clGetPlatformIDs(platformNum, platformIDs, NULL);
	if (errCodeReturn != CL_SUCCESS)
	{
		errCodeOutput(errCodeReturn, "clGetPlatformIDs");
		free(platformIDs);
		return 0;
	}

	cl_uint currDeviceNum = 0;

	for (cl_uint i = 0; i < platformNum; i++)
	{
		cl_uint platformDeviceNum;
		errCodeReturn = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &platformDeviceNum);
		if (errCodeReturn != CL_SUCCESS && errCodeReturn != -1)
		{
			errCodeOutput(errCodeReturn, "clGetDeviceIDs");
			free(platformIDs);
			return 0;
		}

		if (!platformDeviceNum)
			continue;

		cl_device_id* deviceIDs = (cl_device_id*)malloc(sizeof(cl_device_id) * platformDeviceNum);
		if (deviceIDs == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			free(platformIDs);
			free(deviceIDs);
			return 0;
		}

		errCodeReturn = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, platformDeviceNum, deviceIDs, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clGetDeviceIDs");
			free(platformIDs);
			free(deviceIDs);
			return 0;
		}

		for (cl_uint d = 0; d < platformDeviceNum; d++)
		{
			devices[currDeviceNum].ID = deviceIDs[d];

			errCodeReturn = clGetDeviceInfo(deviceIDs[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &devices[currDeviceNum].type, NULL);
			if (errCodeReturn != CL_SUCCESS)
			{
				errCodeOutput(errCodeReturn, "clGetDeviceInfo");
				free(platformIDs);
				free(deviceIDs);
				return 0;
			}

			errCodeReturn = clGetDeviceInfo(deviceIDs[d], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &devices[currDeviceNum].hostUnifiedMem, NULL);
			if (errCodeReturn != CL_SUCCESS)
			{
				errCodeOutput(errCodeReturn, "clGetDeviceInfo");
				free(platformIDs);
				free(deviceIDs);
				return 0;
			}

			currDeviceNum++;
		}

		free(deviceIDs);
	}

	free(platformIDs);
	return 0;
}

void deviceSorting(struct deviceInfo* devices, cl_uint deviceNum)
{
	cl_uint currDeviceNum = 0;

	for (cl_uint i = 0; i < deviceNum; i++)
	{
		if (devices[i].type == CL_DEVICE_TYPE_GPU) {
			if (devices[i].hostUnifiedMem == 0)
			{
				devices[i].sortID = currDeviceNum;
				currDeviceNum++;
			}
		}
	}

	for (cl_uint i = 0; i < deviceNum; i++)
	{
		if (devices[i].type == CL_DEVICE_TYPE_GPU) {
			if (devices[i].hostUnifiedMem == 1)
			{
				devices[i].sortID = currDeviceNum;
				currDeviceNum++;
			}
		}
	}

	for (cl_uint i = 0; i < deviceNum; i++)
	{
		if (devices[i].type == CL_DEVICE_TYPE_CPU)
		{
			devices[i].sortID = currDeviceNum;
			currDeviceNum++;
		}
	}

	for (cl_uint i = 0; i < deviceNum; i++)
	{
		if (devices[i].type != CL_DEVICE_TYPE_GPU && devices[i].type != CL_DEVICE_TYPE_CPU)
		{
			devices[i].sortID = currDeviceNum;
			currDeviceNum++;
		}
	}
}

void deviceSelection(struct deviceInfo* devices, cl_uint deviceNum, cl_device_id* device, unsigned int selectedDeviceID)
{
	for (cl_uint i = 0; i < deviceNum; i++)
	{
		if (devices[i].sortID == selectedDeviceID)
			*device = devices[i].ID;
	}
}

unsigned char buildDefCreation(const char* buildDef, char** buildDefStr, size_t vectorWidth, size_t maxLocalGroupSize)
{
	unsigned int buildDefSize = snprintf(NULL, 0, buildDef, maxLocalGroupSize, vectorWidth, vectorWidth) + 1;
	*buildDefStr = (char*)malloc(sizeof(char) * buildDefSize + 1);
	if (*buildDefStr == NULL)
	{
		fprintf(stderr, "Insufficient memory available!\n");
		return 1;
	}

	if (snprintf(*buildDefStr, buildDefSize, buildDef, maxLocalGroupSize, vectorWidth, vectorWidth) + 1 != buildDefSize)
	{
		fprintf(stderr, "Failed to form build definitions string!\n");
		return 1;
	}

	return 0;
}

unsigned int dimensionAlignment(unsigned int dim, size_t maxLocalGroupSize)
{
	unsigned int alignedDim = (dim / maxLocalGroupSize) * maxLocalGroupSize;
	if (alignedDim < dim)
		alignedDim += maxLocalGroupSize;

	return alignedDim;
}

int main(int argc, char* argv[])
{
	if (argc == 5)
	{
		int selectedDeviceID = atoi(argv[1]);
		const int implementationType = atoi(argv[4]);

		if (1 > implementationType || implementationType > 3)
		{
			fprintf(stderr, "Incorrect implementation type!\n");
			return 1;
		}

		FILE* inputFile = fopen(argv[2], "r");
		if (inputFile == NULL)
		{
			fprintf(stderr, "Input file open error!\n");
			return 1;
		}

		cl_uint platformNum;
		cl_uint deviceNum = getDeviceNumber(&platformNum);
		if (!deviceNum)
		{
			fprintf(stderr, "Number of devices: 0\n");
			fclose(inputFile);
			return 1;
		}

		if (0 > selectedDeviceID || selectedDeviceID >= deviceNum)
			selectedDeviceID = 0;

		struct deviceInfo* devices = (struct deviceInfo*)malloc(sizeof(struct deviceInfo) * deviceNum);
		if (devices == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			fclose(inputFile);
			free(devices);
			return 1;
		}

		if (getDeviceInfo(devices, deviceNum, platformNum))
		{
			fprintf(stderr, "Number of devices: 0\n");
			fclose(inputFile);
			free(devices);
			return 1;
		}

		deviceSorting(devices, deviceNum);

		cl_device_id device;
		deviceSelection(devices, deviceNum, &device, selectedDeviceID);

		free(devices);

		size_t maxLocalGroupSize = 1;
		if (getDeviceNameAndMaxLocalGroupSize(device, &maxLocalGroupSize, implementationType))
		{
			fclose(inputFile);
			return 1;
		}

		struct sizes size;
		if (matrixSizing(inputFile, &size))
		{
			fprintf(stderr, "Invalid matrix sizes!\n");
			fclose(inputFile);
			return 1;
		}

		float* firstMatrix = (float*)malloc(sizeof(float) * size.firstMatrix);
		if (firstMatrix == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			free(firstMatrix);
			fclose(inputFile);
			return 1;
		}

		float* secondMatrix = (float*)malloc(sizeof(float) * size.secondMatrix);
		if (secondMatrix == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			free(firstMatrix);
			free(secondMatrix);
			fclose(inputFile);
			return 1;
		}

		if (readFile(inputFile, firstMatrix, secondMatrix, &size))
		{
			fprintf(stderr, "Invalid file format!\n");
			free(firstMatrix);
			free(secondMatrix);
			fclose(inputFile);
			return 1;
		}

		fclose(inputFile);

		size_t kernelFileSize;
		unsigned char* kernelFileText;

		char* kernelFilePath = "kernel.cl";

		if (implementationType == 2)
			kernelFilePath = "kernelLocalMem.cl";

		if (implementationType == 3)
			kernelFilePath = "kernelVector.cl";

		if (kernelFileProcessing(kernelFilePath, &kernelFileSize, &kernelFileText))
		{
			fprintf(stderr, "Kernel file open error!\n");
			free(firstMatrix);
			free(secondMatrix);
			return 1;
		}

		cl_int errCodeReturn = CL_SUCCESS;
		cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clCreateContext");
			free(firstMatrix);
			free(secondMatrix);
			return 1;
		}

		cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clCreateCommandQueue");
			free(firstMatrix);
			free(secondMatrix);
			clReleaseContext(context);
			return 1;
		}

		cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelFileText, &kernelFileSize, &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clCreateProgramWithSource");
			free(firstMatrix);
			free(secondMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		size_t vectorWidth = 1;
		if (implementationType == 3)
			vectorWidth = 4;

		const char buildDef[] = "-D LSIZE=%uU -D vecWidth=%zu -D floatType=float%zu";
		char* buildDefStr;
		if (buildDefCreation(buildDef, &buildDefStr, vectorWidth, maxLocalGroupSize))
		{
			free(firstMatrix);
			free(secondMatrix);
			free(buildDefStr);
			clFlush(queue);
			clFinish(queue);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clBuildProgram(program, 1, &device, buildDefStr, NULL, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			size_t errLogSize;
			errCodeReturn = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &errLogSize);
			if (errCodeReturn != CL_SUCCESS)
			{
				errCodeOutput(errCodeReturn, "clGetProgramBuildInfo");
				free(firstMatrix);
				free(secondMatrix);
				free(buildDefStr);
				clFlush(queue);
				clFinish(queue);
				clReleaseProgram(program);
				clReleaseCommandQueue(queue);
				clReleaseContext(context);
				return 1;
			}

			unsigned char* errLog = (unsigned char*)malloc(sizeof(char) * errLogSize);
			if (errLog == NULL)
			{
				fprintf(stderr, "Insufficient memory available!\n");
				free(firstMatrix);
				free(secondMatrix);
				free(buildDefStr);
				free(errLog);
				clFlush(queue);
				clFinish(queue);
				clReleaseProgram(program);
				clReleaseCommandQueue(queue);
				clReleaseContext(context);
				return 1;
			}

			errCodeReturn = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, errLogSize, errLog, NULL);
			if (errCodeReturn != CL_SUCCESS)
			{
				errCodeOutput(errCodeReturn, "clGetProgramBuildInfo");
				free(firstMatrix);
				free(secondMatrix);
				free(buildDefStr);
				free(errLog);
				clFlush(queue);
				clFinish(queue);
				clReleaseProgram(program);
				clReleaseCommandQueue(queue);
				clReleaseContext(context);
				return 1;
			}

			fprintf(stderr, "Build log: %s\n", errLog);
			free(firstMatrix);
			free(secondMatrix);
			free(buildDefStr);
			free(errLog);
			clFlush(queue);
			clFinish(queue);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		free(buildDefStr);

		cl_kernel kernel = clCreateKernel(program, "matrixMultiplication", &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "matrixMultiplication");
			free(firstMatrix);
			free(secondMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		float* resultMatrix = (float*)malloc(sizeof(float) * size.resultMatrix);
		if (resultMatrix == NULL)
		{
			fprintf(stderr, "Insufficient memory available!\n");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		unsigned int alignedRowSize = dimensionAlignment(size.rowFirstMatrix, maxLocalGroupSize);
		unsigned int alignedColSize = dimensionAlignment(size.colSecondMatrix, maxLocalGroupSize);
		unsigned int alignedColRowSize = dimensionAlignment(size.colFirstRowSecond, maxLocalGroupSize);

		unsigned int alignedfirstMatrixSize = alignedRowSize * alignedColRowSize;
		unsigned int alignedSecondMatrixSize = alignedColRowSize * alignedColSize;
		unsigned int alignedResultMatrixSize = alignedRowSize * alignedColSize;

		alignedColRowSize /= maxLocalGroupSize;

		cl_mem firstMatrixMem = clCreateBuffer(context, CL_MEM_READ_ONLY, (size_t)(alignedfirstMatrixSize * sizeof(float)), NULL, &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clCreateBuffer");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		cl_mem secondMatrixMem = clCreateBuffer(context, CL_MEM_READ_ONLY, (size_t)(alignedSecondMatrixSize * sizeof(float)), NULL, &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clCreateBuffer");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		cl_mem resultMatrixMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (size_t)(alignedResultMatrixSize * sizeof(float)), NULL, &errCodeReturn);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clCreateBuffer");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseMemObject(secondMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		const cl_uint work_dim = 2;
		size_t global_item_size[2];
		const size_t local_item_size[2] = { maxLocalGroupSize / vectorWidth, maxLocalGroupSize };

		if (implementationType == 1)
		{
			global_item_size[0] = size.colSecondMatrix; 
			global_item_size[1] = size.rowFirstMatrix;
		}
		else
		{
			global_item_size[0] = alignedColSize / vectorWidth;
			global_item_size[1] = alignedRowSize;
		}

		cl_event event_start_transfer, event_end_transfer;

		errCodeReturn = clEnqueueWriteBuffer(queue, firstMatrixMem, CL_FALSE, 0, (size_t)(size.firstMatrix * sizeof(float)), firstMatrix, 0, NULL, &event_start_transfer);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clEnqueueWriteBuffer");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseMemObject(secondMatrixMem);
			clReleaseMemObject(resultMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clEnqueueWriteBuffer(queue, secondMatrixMem, CL_TRUE, 0, (size_t)(size.secondMatrix * sizeof(float)), secondMatrix, 0, NULL, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clEnqueueWriteBuffer");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseMemObject(secondMatrixMem);
			clReleaseMemObject(resultMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clSetKernelArg(kernel, 0, sizeof(cl_mem), &firstMatrixMem);
		errCodeReturn = clSetKernelArg(kernel, 1, sizeof(cl_mem), &secondMatrixMem);
		errCodeReturn = clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultMatrixMem);
		errCodeReturn = clSetKernelArg(kernel, 3, sizeof(cl_uint), &size.colFirstRowSecond);
		errCodeReturn = clSetKernelArg(kernel, 4, sizeof(cl_uint), &size.colSecondMatrix);

		if (implementationType != 1)
		{
			errCodeReturn = clSetKernelArg(kernel, 5, sizeof(cl_uint), &size.rowFirstMatrix);
			errCodeReturn = clSetKernelArg(kernel, 6, sizeof(cl_uint), &alignedColRowSize);
		}

		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clSetKernelArg");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseMemObject(secondMatrixMem);
			clReleaseMemObject(resultMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		cl_event event_kernel;
		cl_ulong kernel_start_time, kernel_end_time;
		cl_ulong transfer_start_time, transfer_end_time;

		if (implementationType == 1)
			errCodeReturn = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_item_size, NULL, 0, NULL, &event_kernel);
		else
			errCodeReturn = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_item_size, local_item_size, 0, NULL, &event_kernel);

		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clEnqueueNDRangeKernel");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseMemObject(secondMatrixMem);
			clReleaseMemObject(resultMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clEnqueueReadBuffer(queue, resultMatrixMem, CL_TRUE, 0, sizeof(float) * size.resultMatrix, resultMatrix, 0, NULL, &event_end_transfer);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clEnqueueReadBuffer");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseMemObject(firstMatrixMem);
			clReleaseMemObject(secondMatrixMem);
			clReleaseMemObject(resultMatrixMem);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clReleaseMemObject(firstMatrixMem);
		errCodeReturn = clReleaseMemObject(secondMatrixMem);
		errCodeReturn = clReleaseMemObject(resultMatrixMem);

		errCodeReturn = clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start_time, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clGetEventProfilingInfo");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end_time, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clGetEventProfilingInfo");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clGetEventProfilingInfo(event_start_transfer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &transfer_start_time, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clGetEventProfilingInfo");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clGetEventProfilingInfo(event_end_transfer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &transfer_end_time, NULL);
		if (errCodeReturn != CL_SUCCESS)
		{
			errCodeOutput(errCodeReturn, "clGetEventProfilingInfo");
			free(firstMatrix);
			free(secondMatrix);
			free(resultMatrix);
			clFlush(queue);
			clFinish(queue);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 1;
		}

		errCodeReturn = clFlush(queue);
		errCodeReturn = clFinish(queue);
		errCodeReturn = clReleaseKernel(kernel);
		errCodeReturn = clReleaseProgram(program);
		errCodeReturn = clReleaseCommandQueue(queue);
		errCodeReturn = clReleaseContext(context);

		free(firstMatrix);
		free(secondMatrix);

		double kernel_runtime = (kernel_end_time - kernel_start_time) / 1000000.0;
		double transfer_runtime = (transfer_end_time - transfer_start_time) / 1000000.0;

		printf("Time: %g\t%g\n", kernel_runtime, transfer_runtime);

		if (implementationType == 2)
			printf("LOCAL_WORK_SIZE[%i, %i]\n", maxLocalGroupSize, maxLocalGroupSize);

		if (implementationType == 3)
		{
			printf("LOCAL_WORK_SIZE[%i, %i]\n", maxLocalGroupSize, maxLocalGroupSize / (int)vectorWidth);
			printf("WI_WORK %i\n", (int)vectorWidth);
		}

		FILE* outputFile = fopen(argv[3], "wb");
		if (outputFile == NULL)
		{
			fprintf(stderr, "Output file open error!\n");
			free(resultMatrix);
			return 1;
		}

		FILE* outputFile = fopen(argv[3], "wb");
		if (outputFile == NULL)
		{
			fprintf(stderr, "Output file open error!\n");
			free(resultMatrix);
			return 1;
		}

		if (writeFile(outputFile, resultMatrix, &size))
		{
			fprintf(stderr, "File write error!\n");
			free(resultMatrix);
			fclose(outputFile);
			return 1;
		}

		fclose(outputFile);
		free(resultMatrix);
	}
	else
	{
		fprintf(stderr, "Wrong number of arguments!\n");
		return 1;
	}
}
