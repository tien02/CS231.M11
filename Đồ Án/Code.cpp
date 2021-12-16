#include <iostream>
#include <opencv2/dpm.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>


using namespace std;
using namespace cv;
using namespace cv::dpm;
using namespace cv::dnn;

// DPM Model Path
const string model_path = "path2xml\\inriaperson.xml";

void ResizeBoxes(cv::Rect& box) 
{
	box.x += cvRound(box.width * 0.1);
	box.width = cvRound(box.width * 0.8);
	box.y += cvRound(box.height * 0.06);
	box.height = cvRound(box.height * 0.8);
}

int main(int argc, char** argv)
{
	// Image Path
	string path = "Images/camera1.png";
	Mat org_img = imread(path);
	if (!org_img.data)
	{
		cout << "\n====No image with path " << path << ".Try again!===" << endl;
		return -1;
	}

	//Resize Image
	Mat img;
	resize(org_img, img, Size(480, 360));

	Mat hog_result = img.clone();
	Mat new_img = img.clone();
	Mat dpm_result = img.clone();

	///==================Dalal & Triggs Model=====================//
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	vector<Rect> detections;
	hog.detectMultiScale(hog_result, detections, -1, Size(10, 10), Size(32, 32), 1.2, 2);

	for (auto& detection : detections) {
		ResizeBoxes(detection);
		rectangle(hog_result, detection.tl(), detection.br(), cv::Scalar(0, 255, 0), 2);
	}


	//------------------Deformable Part Model---------------------//

	const char* keys =
	{
		"{@model_path    | | Path of the DPM cascade model}"
	};
			
	CommandLineParser parser(argc, argv, keys);
	//string model_path(parser.get<string>(0));
			
	if (model_path.empty())
	{
		cout << "Please Enter Model Path First!" << endl;
		return -1;
	}
			
	cv::Ptr<DPMDetector> detector = \
		DPMDetector::create(vector<string>(1, model_path));

	vector<DPMDetector::ObjectDetection> ds;
	detector->detect(new_img, ds);

	vector<Rect> currentbboxes;
	vector<float> currentscores;
	vector<int> nmsclassID;

	for (unsigned int i = 0; i < ds.size(); i++)
	{
		currentbboxes.push_back(ds[i].rect);
		if (ds[i].score >= 0)
		{
			currentscores.push_back(ds[i].score);
		}
		else
		{
			currentscores.push_back(-1 * ds[i].score);
		}
	}

	NMSBoxes(currentbboxes, currentscores, 0, 0.4, nmsclassID);

	for (unsigned int i = 0; i < nmsclassID.size(); i++)
	{
			rectangle(dpm_result, ds[nmsclassID[i]].rect, cv::Scalar(0, 255, 0), 2);
	}

	/*for (unsigned int i = 0; i < ds.size(); i++)
	{
		if (ds[i].score > 0)
		{
			rectangle(dpm_result, ds[i].rect, cv::Scalar(0, 255, 0), 2);
		}
	}*/

	//Save Image
	//imwrite("Results\\camera1.png", dpm_result);

	// Show Image
	imshow("HOG", hog_result);
	imshow("DPM", dpm_result);
	waitKey(0);
}
