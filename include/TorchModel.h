#ifndef TORCHMODEL_H
#define TORCHMODEL_H
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include "string"
#include <opencv2/opencv.hpp>
	
class TorchModel 
{
    public:
         TorchModel();
         ~TorchModel();
	 bool loadModel(const std::string &modelPath);
	 torch::Tensor predict(const  std::vector<torch::Tensor> &input);
	 void changeDevice(const torch::DeviceType &deviceSet, const int &index);
	 torch::Tensor convertToTensor(const cv::Mat& img, bool normalize=false, bool color=true);
	 torch::Tensor convertListToTensor(std::list<cv::Mat>& imgs);
	 torch::Tensor predictTuple(const std::vector<torch::Tensor>& input);
	 cv::Mat convertToMat(const torch::Tensor& output, bool isFloat, bool permute,bool bgr, bool color);
	 cv::Mat loadMat(const std::string file, bool grey, bool rgb);
	 torch::jit::script::Module model;
	 //Default device is CUDA, if avail
	 torch::Device device = torch::kCUDA;
 }; 
#endif
