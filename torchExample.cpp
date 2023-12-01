#include "CRAFT.h"
#include "TorchModel.h"
#include <torch/torch.h>
#include <chrono>
#include "CRNN.h"
using namespace torch::indexing;
int main()
{
	torch::NoGradGuard no_grad_guard;
	c10::InferenceMode guard;
	// Both are inherited from TorchModel objects
	CRNNModel recognition;
	CraftModel detection;
	// Can optionally set number of threads, set to mimic 4 threads like Tesseract
	cv::setNumThreads(4);
	torch::set_num_threads(4);
	
	std::string det = "../models/CRAFT-detector.pt";
	std::string rec = "../models/traced-recog.pt";
	
	// Set your input image here!
	std::string filePath = "../test.jpg";

	auto startModel = std::chrono::steady_clock::now();
	// Always check the model was loaded successully
	auto check_rec = recognition.loadModel(rec.c_str());
	auto check_det = detection.loadModel(det.c_str());
	auto endModel = std::chrono::steady_clock::now();

	auto diff = endModel - startModel;
	std::cout << "MODEL TIME " << std::chrono::duration <double, std::milli>(diff).count() << " ms" << std::endl;

	//CHECK IF BOTH MODEL LOADED SUCESSFULLY
	if (check_rec && check_det) 
	{
		int runs = 1;
		// Load in image into openCV Mat (bW or color)
		cv::Mat matInput = detection.loadMat(filePath, false, true).clone();
		// resizes input if we need to
		HeatMapRatio processed = detection.resizeAspect(matInput);
		cv::Mat clone = processed.img.clone();
		cv::Mat grey = processed.img.clone();
		grey.convertTo(grey, CV_8UC1);
		cv::cvtColor(grey,grey, cv::COLOR_BGR2GRAY);
		torch::Tensor tempTensor = detection.convertToTensor(grey.clone(), true, false).squeeze(0);
		clone.convertTo(clone, CV_8UC3);
		for (int i = 0; i < runs; i++)
		{
			
			torch::Tensor input = detection.preProcess(processed.img.clone());
			auto ss = std::chrono::high_resolution_clock::now();
			// use custom algorithm for bounding box merging
			std::vector<BoundingBox> dets = detection.runDetector(input,true);
			int maxWidth;
			std::vector<TextResult> results = recognition.recognize(dets, grey,maxWidth);
			auto ee = std::chrono::high_resolution_clock::now();
			auto difff = ee - ss;
			int count = 0;
			for (auto x : dets)
			{
				rectangle(clone, x.topLeft, x.bottomRight, cv::Scalar(0, 255, 0));
				putText(clone, std::to_string(count), (x.bottomRight + x.topLeft)/2, cv::FONT_HERSHEY_COMPLEX, .6, cv::Scalar(100,0, 255));
				count++;

			}
			for (auto& result : results)
			{
				std::cout << "LOCATION: " << result.coords.topLeft << " " << result.coords.bottomRight << std::endl;
				std::cout << "TEXT: " << result.text << std::endl;
				std::cout << "CONFIDENCE " << result.confidence << std::endl;
				std::cout << "################################################" << std::endl;
			}
			

			cv::imwrite("../output-heatmap.jpg", clone);
			std::cout << "TOTAL INFERENCE TIME " << std::chrono::duration <double, std::milli>(difff).count() << " ms" << std::endl;


		}

	}
	return 0;
}
