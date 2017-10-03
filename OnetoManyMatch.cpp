#include "utils.h"
#include "TransformColor.h"
#include "FindROI.h"
using namespace cv;
void OnetoManyMatch(Mat templateImage, Mat targetImage)
{
	Mat blurI1, blurI2;
	GaussianBlur(templateImage, blurI1, cv::Size(3, 3), 0.8, 0.8);
	GaussianBlur(targetImage, blurI2, cv::Size(3, 3), 0.8, 0.8);

	namedWindow("blurtemplateimage", CV_WINDOW_NORMAL);
	imshow("blurtemplateimage", blurI1);
	waitKey(0);
	namedWindow("blurtargetimage", CV_WINDOW_NORMAL);
	imshow("blurtargetimage", blurI2);
	waitKey(0);


	ROI* rois;
	int numROI;
	findroi(rois, numROI, blurI1, blurI2);
}