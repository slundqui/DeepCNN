#ifndef TESTUTILS_HPP_ 
#define TESTUTILS_HPP_ 

#include <cmath>

#include <Column.hpp>
#include <connections/Convolution.hpp>
#include <layers/MatInput.hpp>
#include <layers/BaseCostFunction.hpp>

bool gradientCheck(Column* myCol, Convolution* checkConn, MatInput* input, MatInput* gt, BaseCostFunction* cost, float tolerance, float* h_actualWGrad, float* h_actualBGrad);
#endif

