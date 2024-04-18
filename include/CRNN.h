#ifndef CRNN_H
#define CRNN_H
#include <torch/script.h>
#include <torch/torch.h>
#include "string"
#include "TorchModel.h"
#include "CRAFT.h"
#include <opencv2/opencv.hpp>
struct TextResult
{
	std::string text;
	float confidence;
	BoundingBox coords;
};

class CRNNModel : public TorchModel {

public:

	CRNNModel();
	std::vector<TextResult> recognize(std::vector<BoundingBox>& dets, cv::Mat& img, int& maxWidth);
	torch::Tensor preProcess(cv::Mat& det);
	torch::Tensor normalizePad(cv::Mat& processed, int minWidth);
	std::string greedyDecode(torch::Tensor& input, int size);
	//stores the last computed ratio (resize/rescale) from input image. 
	float ratio;
	std::vector<char> characters;
};
#endif
