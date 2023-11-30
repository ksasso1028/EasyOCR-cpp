# EasyOCR-cpp
![alt text](https://github.com/ksasso1028/EasyOCR-cpp/blob/main/output-heatmap.jpg)
### Custom C++ implementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)
### Built and tested on Windows 11, libtorch 1.13+cpu and OpenCV 4.6

This C++ project implements the pre/post processing to run a OCR pipeline consisting of a text detector [CRAFT](https://arxiv.org/abs/1904.01941), and a CRNN based text recognizer. Unlike the EasyOCR python which is API based, this repo provides a set of classes to show how you can integrate OCR in any C++ program for maximum flexibility. The torchExample.cpp main program highlights how to utilize all elements of the EasyOCR-cpp pipeline. Because a test program is only provided, make sure to configure your input image within torchExample.cpp if you only plan to utilize the test program


Libtorch is being utilized with an in-house class I ussualy use for C++ inference [TorchModel](https://github.com/ksasso1028/EasyOCR-cpp/blob/main/src/TorchModel.cpp), and OpenCV for the pre/post processing steps.
The TorchModel class can easily adapted to run inference on most Pytorch models converted to [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html). Provides some handy functions to pre process opencv::Mat and handle device usage (GPU,CPU)

Some features that have yet to be implemented:
-beam search, only implemented greedy decoding
-.txt/.pdf output
-exact bounding box merge alg from EasyOCR, opted for custom one which is less complex
-support for other languages, atm only english is supported.
-not sure how well linux is supported, any feedback is appreciated


### If you would like to support feel free to make a PR, or a issue if you are having trouble.

## Steps to run 

### Dependencies
Click to Download libtorch - > [download](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.13.1%2Bcpu.zip)

Use OpenCV Windows installer and unzip (v4.6) - > [OpenCV libs](https://opencv.org/releases/)

Make sure to change the location in the Makefile for OpenCV to point to your OpenCV build dir in the [Makefile](https://github.com/ksasso1028/EasyOCR-cpp/blob/e9311ee3f45b59c2709be3a98a04b48c215a845b/CMakeLists.txt#L7)

Create a build directory within the repo, cd to it and run
```
cmake -DCMAKE_PREFIX_PATH=<absolute-path-to-libtorch-folder> ..
```


This will generate a solution within the build folder you can open up in Visual Studio. **Make sure to use the Release config when building**

### Running

Configure your input image  [here](https://github.com/ksasso1028/EasyOCR-cpp/blob/e9311ee3f45b59c2709be3a98a04b48c215a845b/torchExample.cpp#L25). Currently the test program is using the test.jpg which comes in the repo.

Launch from command-line, or within Visual Studio after building.

**Since its designed to be used in a C++ program, text is not being written to disk at*** An output image will be generated in the main repo dir containing an annotated version of the input image with detection bounding boxes


