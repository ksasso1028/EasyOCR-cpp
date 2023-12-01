#include "CRNN.h"
#include <tuple>
using namespace torch::indexing;


CRNNModel::CRNNModel() : TorchModel()
{
	for (int i = 0;i<this->e.size();i++)
	{
		this->english[(i+1)] = std::string(1,this->e[i]);
	}
	// Set the blank token
	std::string blank = "[blank]";
	this->english[0]= blank;
	this->e.insert(this->e.begin(), ' ');
	for (auto x : this->english)
	{
		std::cout << x.first << "  " << x.second << std::endl;
	}
	
}

float resizeComputeRatio(cv::Mat& img, int modelHeight)
{
	float ratio = float(img.cols) / float(img.rows);
	if (ratio < 1.0)
	{
		ratio = 1.0 / ratio;
		cv::resize(img, img, cv::Size(modelHeight, int(modelHeight * ratio) ));
	}
	else
	{
		
		cv::resize(img, img, cv::Size(int(modelHeight * ratio),modelHeight ));
		
	}
	
	return ratio;
}

// Greedy decoding
std::string CRNNModel::greedyDecode(torch::Tensor& input, int size)
{
	int length = size;
	std::vector<int> ignoreList = { 0 };
	torch::Tensor t = input.slice(0, 0, input.size(0));
	auto a = torch::cat({ torch::tensor({true}), ~(t.slice(0,0, -1) == t.slice(0,1).flatten()) }, 0);
	auto b = ~(t.unsqueeze(1) == torch::tensor(ignoreList).unsqueeze(0)).all(1);
	auto c = a & b;
	auto indices = c.nonzero();
	auto result = t.index_select(0, indices.flatten());
	std::vector<char> extracted;
	for (int i = 0; i < result.size(0); i++) {
		int index = result[i].item<int>();
		if (index >= 0 && index < this->e.size()) {
			extracted.push_back(this->e[index]);
		}
	}
	// Join the extracted characters into a single string
	std::string text(extracted.begin(), extracted.end());
	return text;
}

//still need to implement beam search


torch::Tensor CRNNModel::preProcess(cv::Mat& det) 
{
	// Default model height used in easyOCR
	float ratio = resizeComputeRatio(det, 64);
	double alpha = 1.28;
	double beta = 0;
	//cv::equalizeHist(det, det);
	//det.convertTo(det, -1, alpha, beta);

	//at least 128 in length
	auto processedTensor = this->normalizePad(det, 256);
	return processedTensor;
}

std::vector<TextResult> CRNNModel::recognize(std::vector<BoundingBox>& dets, cv::Mat& img, int &maxWidth)
{
	// returns max width for padding and resize
	std::vector<torch::Tensor> processed;
	float maxRatio = 0;
	std::vector<TextResult> results;
	for (auto& x : dets)
	{
		TextResult res;
		cv::Mat det = img(cv::Rect(x.topLeft.x, x.topLeft.y, (x.bottomRight.x - x.topLeft.x), (x.bottomRight.y - x.topLeft.y))).clone();
		if (det.rows < 5)
			continue;
		
		torch::Tensor processedTensor = this->preProcess(det);
		std::vector<torch::Tensor>  input{ processedTensor.unsqueeze(0) };

		auto ss = std::chrono::high_resolution_clock::now();
		torch::Tensor output = this->predict(input);
		auto ee = std::chrono::high_resolution_clock::now();
		auto difff = ee - ss;
		//std::cout << "TOTAL INFERENCE RECORNGITON TIME " << std::chrono::duration <double, std::milli>(difff).count() << " ms" << std::endl;
		
		//post process and decode
		auto confidence = torch::softmax(output, 2);
		auto norm = confidence.sum(2);
		auto prob = (confidence / norm.unsqueeze(2));
		torch::Tensor predIndex;
		std::tie(std::ignore, predIndex) = prob.max(2);
		predIndex = predIndex.view({ -1 });
		std::string text = this->greedyDecode(predIndex, predIndex.size(0));
		res.text = text;
		res.confidence = *prob.data<float>();
		res.coords = x;
		results.push_back(res);
		processed.push_back(processedTensor);
	}
	// 64 was model height used in easyOCR
	float maxW = float(ratio * 64);
	return results;
}

torch::Tensor CRNNModel::normalizePad(cv::Mat& processed, int maxWidth) 
{
	std::vector<torch::Tensor> input;
	auto converted = this->convertToTensor(processed.clone(), true, false).squeeze(0);
	torch::Tensor pad = torch::zeros({ 1,converted.size(1),maxWidth});
	converted = (converted - (.5  ) / (.5  ));
	if (maxWidth > converted.size(2)) 
	{
		pad.narrow(2, 0, converted.size(2)).copy_(converted.detach());
		auto padded = this->convertToMat(converted, true, true, false, false);
		//cv::imwrite("Padded.jpg", padded);
		converted = pad.clone();	
	}
	int width = converted.size(2);
	return converted;
}
