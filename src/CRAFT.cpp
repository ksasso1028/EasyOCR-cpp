#include "CRAFT.h"
#include <algorithm>

using namespace torch::indexing;


HeatMapRatio CraftModel::resizeAspect(cv::Mat&  img)
{
	HeatMapRatio output;
	try {
		//cv::resize(rimg, rimg, cv::Size(col, row));
		cv::Mat processed;
		int channels = img.channels();
		float height = img.rows;
		float width = img.cols;
		float targetSize = std::max(height, width);
		const int canvasSize = 2560;
		int targetH, targetW;
		if (targetSize > canvasSize)
		{
			targetSize = canvasSize;
		}
		float ratio = targetSize / std::max(height, width);
		std::cout << "RATIO " << ratio;
		targetH = int(height * ratio);
		targetW = int(width * ratio);
		cv::resize(img, img, cv::Size(targetW, targetH));

		int h32 = targetH;
		int w32 = targetW;
		if (targetH % 32 != 0)
		{
			h32 = targetH + (32 - targetH % 32);
		}

		if (targetW % 32 != 0)
		{
			w32 = targetW + (32 - targetW % 32);
		}

		cv::Mat resized = cv::Mat::zeros(h32, w32, CV_32FC3);
		std::cout << resized.type() << std::endl;
		std::cout << img.type() << std::endl;
		cv::Range colRange = cv::Range(0, cv::min(resized.cols, img.cols)); //select maximum allowed cols
		cv::Range rowRange = cv::Range(0, cv::min(resized.rows, img.rows)); //select maximum allowed cols
		img(rowRange, colRange).clone().copyTo(resized(rowRange, colRange));


		cv::Size heatMapSize = cv::Size(int(targetW / 2), int(targetH / 2));
		output.ratio = ratio;
		output.heatMapSize = heatMapSize;
		output.img = resized;
		this->ratio = ratio;
		
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	return output;
}



std::vector<BoundingBox> CraftModel::mergeBoundingBoxes(std::vector<BoundingBox>& dets, float distanceThresh, int height, int width)
{	

	// represents how much we change the top left Y
	std::sort(dets.begin(), dets.end(), boxSorter());
	//return dets;


	bool merge = NULL;
	std::vector<BoundingBox> merged;
	//  track box index
	int newTopLeft;
	bool firstRun = true;
	for (int i = 0; i < dets.size(); i++)
	{
		cv::Point newBottomRight = dets[i].bottomRight;
		
		int minY = 0;
		int maxY = 0;
		if (i < dets.size())
		{
			float x = dets[i].bottomRight.x;
			float xPrime = dets[i + 1].topLeft.x;
			float ratio = x / xPrime;
			bool canMerge;
			bool isNegative = false;
			if (x - xPrime < 0) isNegative = true;
			//std::cout << "RATIO = " << ratio << std::endl;
			float w= dets[i].bottomRight.x - dets[i].topLeft.x;
			float h = dets[i].bottomRight.y - dets[i].topLeft.y;
			if (width > 5 * height)
			{
				// box is a line, skip merging
				canMerge = false;
			}
			//merge box, store point
			
			if (ratio > distanceThresh && ratio < 1.4 && std::abs(dets[i].bottomRight.y - dets[i+1].bottomRight.y) < 20)
			{
				newBottomRight = dets[i + 1].bottomRight;
				if (dets[i + 1].topLeft.y < dets[i].topLeft.y)
				{
					if (minY > dets[i + 1].topLeft.y)
					{
						minY = dets[i + 1].topLeft.y;
					}
				}
				else {
					if (dets[i + 1].bottomRight.y > maxY)
					{
						maxY = dets[i + 1].bottomRight.y;
					}
					
				}
				canMerge = true;
			}

			else
			{
				newBottomRight = dets[i].bottomRight;
				canMerge = false;

			}

			if (firstRun)
			{
				// first box, we need to continue to flow into other logic
				merge = canMerge;
				// store first
				newTopLeft = i;
				firstRun = false;
			}
			else
			{
				merge = canMerge;
				
	
			}

			// other box is too far
			if (!merge)
			{
				BoundingBox newBox;
				newBox.topLeft = dets[newTopLeft].topLeft;
				newBox.topLeft.y -= minY;
				//margin
				newBox.topLeft.y *= .998;
				newBox.topLeft.x *= .998;
				if (newBox.topLeft.y < 0)
				{
					newBox.topLeft.y = 0;
				}

				if (newBox.topLeft.x < 0)
				{
					newBox.topLeft.x = 0;
				}
				newBox.topLeft.x = int(newBox.topLeft.x);
				newBox.topLeft.y =  int(newBox.topLeft.y);

				newBox.bottomRight = newBottomRight;
				newBox.bottomRight.y += maxY;
				// margin
				newBox.bottomRight.y *= 1.003;
				newBox.bottomRight.x *= 1.003;

				if (newBox.bottomRight.y > height)
				{
					newBox.bottomRight.y = height-1;
				}

				if (newBox.bottomRight.x > width)
				{
					newBox.bottomRight.x = width-5;
				}
				
				newBox.bottomRight.x = int(newBox.bottomRight.x);
				newBox.bottomRight.y =  int(newBox.bottomRight.y);

				
				merged.push_back(newBox);
				// move top left box to next box
				std::cout << " TOP LEFT  " << dets[newTopLeft].topLeft << std::endl;
				std::cout << " BOTTOM RIGHT POINT  " << newBottomRight << std::endl;
				newTopLeft = i + 1;
				//std::cout << " CANNOT MERGE! " << std::endl;
				minY = 0;
				maxY = 0;


			}
		}
	}
	return merged;
}

std::vector<BoundingBox> CraftModel::getBoundingBoxes(torch::Tensor &input, torch::Tensor& output, float textThresh, float linkThresh, float lowText)
{
	std::vector<BoundingBox> detBoxes;
	cv::Mat linkMap = this->convertToMat(output.select(2, 0).unsqueeze(0).clone(),true, true, false, false).clone();
	cv::Mat textMap = this->convertToMat(output.select(2, 1).unsqueeze(0).clone(), true, true, false, false).clone();
	auto tempTextMap = output.select(2, 1).unsqueeze(0).clone();
	//std::cout << linkMap;
	int r = linkMap.rows;
	int c = linkMap.cols;
	cv::Mat linkScore, textScore;
	cv::threshold(linkMap, linkScore, (linkThresh * 255), 255, 0);
	cv::threshold(textMap, textScore, (lowText * 255), 255, 0);
	cv::Mat outputScore = linkScore.clone() + textScore.clone();
	
	cv::min(cv::max(outputScore, 0), 255, outputScore);
	outputScore.convertTo(outputScore,CV_8UC3);
	
	cv::Mat labels, stats, centroids;
	std::vector<int> mapper;
	int numLabels = cv::connectedComponentsWithStats(outputScore, labels, stats, centroids, 4, CV_32S);

	//std::cout << labels << std::endl;
	std::cout << "stats.size()=" << stats.size() << std::endl;
	//std::cout << centroids << std::endl;
	for (int i = 1; i < numLabels; i++)
	{
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		//std::cout << area << std::endl;
		if (area < 10)
			continue;
	
		cv::Mat segMap = cv::Mat::zeros(textMap.size(), CV_8UC1);
		cv::Mat mask = (labels == i);

		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(textMap, &minVal, &maxVal, &minLoc, &maxLoc,mask);
		mapper.push_back(i);

		segMap.setTo(255, labels == i);
		cv::Mat linkMask = (linkScore == 1) & (textScore == 0);
		segMap.setTo(0, linkMask);

		int x = stats.at<int>(i, cv::CC_STAT_LEFT);
		int y = stats.at<int>(i, cv::CC_STAT_TOP);
		int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		int niter = int(sqrt(area * cv::min(w, h) / (w * h)) * 2);
		int sx = x - niter, ex = x + w + niter + 1, sy = y - niter, ey = y + h + niter + 1;
		// boundary check
		if (sx < 0) sx = 0;
		if (sy < 0) sy = 0;
		if (ex >= c) ex = c;
		if (ey >= r) ey = r;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));
	    cv::dilate(segMap(cv::Range(sy, ey), cv::Range(sx, ex)), segMap(cv::Range(sy, ey), cv::Range(sx, ex)), kernel);
		//std::cout << "SEGMAP SHAPE: " << segMap.size() << std::endl;

		// make box
		cv::Mat nonZeroCoords;
		cv::findNonZero(segMap, nonZeroCoords);

		cv::Mat npContours = nonZeroCoords.reshape(2, nonZeroCoords.total()).t();
		cv::Mat boxMat;
		cv::RotatedRect rectangle = cv::minAreaRect(npContours);
		cv::boxPoints(rectangle, boxMat);
		cv::Mat box = boxMat;
		BoundingBox detection;


		std::vector<cv::Point> points;
		for (int i = 0; i < 4; i++)
		{	
			float colVal;
			float rowVal = box.at<float>(i, 0);
			for (int j = 0; j < 1; j++)
			{
				colVal = box.at<float>(i, 1);
			}
			cv::Point point(rowVal, colVal );
			//std::cout << "POINT  " << point << std::endl;
			points.push_back(point);
		}
		std::sort(points.begin(), points.end(), pointSorter());
		//std::cout << "POINTS ORDER " << points << std::endl;
		detection.topLeft.x = (points[0].x * 2);
		detection.topLeft.y = (points[0].y * 2);
		detection.bottomRight.x = (points[3].x * 2);
		detection.bottomRight.y = (points[3].y * 2);
		if (detection.bottomRight.y < detection.topLeft.y)
		{
			std::cout << "ALIGNEMENT ISSUE" << std::endl;
		}
		// align diamond-shape
		float w1 = cv::norm(box.row(0) - box.row(1));
		float h1 = cv::norm(box.row(1) - box.row(2));
		float box_ratio = std::max(w1, h1) / (std::min(w1, h1) + 1e-5);
		/*
		if (std::abs(1 - box_ratio) <= 0.1) {
			double l, r, t, b;
			cv::minMaxLoc(np_contours.col(0), &l, &r);
			cv::minMaxLoc(np_contours.col(1), &t, &b);
			//box = (cv::Mat_<float>(4, 2) << l, t, r, t, r, b, l, b);
			detection.topLeft.x = int(l) * 2;
			detection.topLeft.y = int(t) * 2;
			detection.bottomRight.x = int(r) * 2;
			detection.bottomRight.y = int(b) * 2;
			
		}
		*/
		detBoxes.push_back(detection);
		//std::cout << "BOUNDING BOX: " << box << std::endl;

	}
	//std::cout<< "NUMBER OF COMPONENTs we stored" << mapper.size() << std::endl;
	// # uncomment to see raw output written to disk
	//cv::imwrite("output-heatmap.jpg", outputScore);


	return detBoxes;



}

cv::Mat CraftModel::normalize(cv::Mat &img)
{
	std::vector<cv::Mat> channels(3);
	cv::Mat output;
	// split img:
	split(img, channels);
	//mean = (0.485, 0.456, 0.406)
	//variance = (0.229, 0.224, 0.225)
	channels[0] = (channels[0] - (.485 * 255)) / (.229 * 255);
	channels[1] = (channels[1] - (.456 * 255)) / (.224 * 255);
	channels[2] = (channels[2] - (.406 * 255)) / (.225 * 255);
	merge(channels, output);

	return output;
}

torch::Tensor CraftModel::preProcess(cv::Mat & matInput)
{

	//Normalize the input using mean + std values from easyOCR
	matInput = this->normalize(matInput.clone()).clone();
	//Convert final input into a torch::Tensor from a cv::Mat
	torch::Tensor input = this->convertToTensor(matInput.clone()).clone();
	return input;
}

std::vector<BoundingBox> CraftModel::runDetector(torch::Tensor& input, bool merge)
{
	int height = input.size(2);
	int width = input.size(3);
	//std::cout << " HEIGHT IS " << height << std::endl;
	//std::cout << " WIDTH IS " << width << std::endl;
	std::vector<torch::Tensor> detInput = { input.clone() };
	//std::cout << input << std::endl;
	auto output = this->predict(detInput).squeeze().detach().clone();
	//std::cout << output << std::endl;
	auto ss = std::chrono::high_resolution_clock::now();
	auto detections = this->getBoundingBoxes(input.clone(),output.clone());
	//custom bounding box merging
	if (merge)
		detections = this->mergeBoundingBoxes(detections, .97, height, width);

	auto ee = std::chrono::high_resolution_clock::now();
	auto difff = ee - ss;
	std::cout << "TOTAL preprocessing TIME " << std::chrono::duration <double, std::milli>(difff).count() << " ms" << std::endl;
	return detections;
}

