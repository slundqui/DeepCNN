#ifndef INCLUDES_HPP_ 
#define INCLUDES_HPP_ 
#include <iostream>
#include <stdio.h>
#include <vector>
#include <matio.h>
#include <cudnn.h>
#include <cstdlib>
#include <cassert>
#include <string>
#include <cuda_runtime.h>

#define DEBUG 0

//Return commands
#define SUCCESS 0
#define UNDEFINED_PARAMS 1
#define INVALID_DATA_ADDITION 2
#define FILEIO_ERROR 3
#define ERROR 255
#endif
