
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
	std::vector<TextResult> CRNNModel::recognize(std::vector<BoundingBox>& dets, cv::Mat& img, int& maxWidth);
	torch::Tensor CRNNModel::preProcess(cv::Mat& det);
	
	torch::Tensor CRNNModel::normalizePad(cv::Mat& processed, int minWidth);
	std::string CRNNModel::greedyDecode(torch::Tensor& input, int size);
	//stores the last computed ratio (resize/rescale) from input image. 
	float ratio;
	// need to support more languages.. english only for now
	std::unordered_map<int,std::string> english;
	// lol
	std::vector<char> e{ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[','\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '€', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
};
#endif
