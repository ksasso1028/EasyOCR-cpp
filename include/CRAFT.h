#ifndef CRAFT_H
#define CRAFT_H
#include <torch/script.h>
#include <torch/torch.h>
#include "string"
#include "TorchModel.h"
#include <opencv2/opencv.hpp>

struct HeatMapRatio {
	cv::Mat img;
	cv::Size heatMapSize;
	float ratio;
};
struct BoundingBox {
	cv::Point topLeft;
	cv::Point bottomRight;
};
struct boxSorter {
	bool operator()(const BoundingBox& a, const BoundingBox& b) {
		// Check if the boxes are on the same row
		if (std::abs(a.bottomRight.y - b.bottomRight.y) < 7) {

			return a.bottomRight.x < b.bottomRight.x;
		}
		// If the boxes are not on the same row, sort by their y-coordinate
		else {
			return a.bottomRight.y < b.bottomRight.y;
		}
	}
};

struct pointSorter {
	bool operator()(const cv::Point& a, const cv::Point& b) {
		int sumA = a.x + a.y;
		int sumB = b.x + b.y;
		return sumA < sumB;
	}
};

class CraftModel: public TorchModel{

public:
	HeatMapRatio resizeAspect(cv::Mat& img);
	cv::Mat normalize(const cv::Mat & img);
	std::vector<BoundingBox> getBoundingBoxes(const torch::Tensor &input, const torch::Tensor& output, float textThresh = .7, float linkThresh = .4, float lowText = .4);
	torch::Tensor preProcess(const cv::Mat & matInput);
	std::vector<BoundingBox> mergeBoundingBoxes(std::vector<BoundingBox>& dets, float distanceThresh, int height, int width);
	std::vector<BoundingBox> runDetector(torch::Tensor& input, bool merge);
	// stores the last computed ratio (resize/rescale) from input image. 
	float ratio;
};
#endif
