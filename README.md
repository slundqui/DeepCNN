# DeepCNN
A repository for building a GPU accelerated DeepCNN framework

##Prerequisites
- MatIO (a library for writing and reading Matlab .mat files)
- Cuda 7.5
- CuDNN v3
- CMake
- G++
- GTest (for system tests)
- CImg (for reading images)
- Git
- Octave

##Build
- Clone the repository from github: 
```
git clone https://github.com/slundqui/DeepCNN.git
```
- Run CMake
```
ccmake .
```
- Fill out appropriate fields
- Press g
- Press c
- Compile
```
make
```

##Run
Download CIFAR-10 database.
Run script for formatting mat file, editing file to correct path
```
octave <repoBaseDir>/scripts/formatCifar.m
```

Executables in <repoBaseDir>/bin:
- test: The testing suite for the toolkit
- cifarTrain: Executable for training
- cifarTest: Executable for testing, loading weights from cifarTrain
- cifarTest_IO: Executable for testing using weights from epoch 15. Writes out each estimate and ground truth for confusion matrix generation.
	
