#include "utils.h"
using namespace cv;
Mat transformcolor(Mat inputimage)
{
	int h = inputimage.rows;
	int w = inputimage.cols;
	Mat outputimage(h, w, CV_64FC3);
	unsigned char * inputptr = inputimage.data;
	double * outputptr = (double *)outputimage.data;
	for (int y = 0; y < h; y++)
	{
		unsigned char * tmpinputptr = inputptr + 3 *y* w;
		double * tmpoutputptr = outputptr + y* w *3;
		for (int x = 0; x < w; x++)
		{
			double B = *(tmpinputptr++);
			double G = *(tmpinputptr++);
			double R = *(tmpinputptr++);
			double Max, Min;
			Max = B > G ? B : G;
			Max = Max > R ? Max : R;
			Min = B < G ? B : G;
			Min = Min < R ? Min : R;
			double H;
			double range = Max - Min + 0.1;
			if (Max == R)
			{
				H = 60 * ((G - B) / range);
			}
			else if (Max == G)
			{
				H = 120 + 60 * ((B - R) / range);
			}
			else
			{
				H = 240 + 60 * ((R - G) / range);
			}
			H = H < 0 ? (H + 360) : H;
			//*(tmpoutputptr++) = H;
			//double O1 = (0.5f    * (255 + G - R));
			//double O2 = (0.25f    * (510 + G + R - 2 * B));
			//double O1_ = (O1 / (O1 + O2));
			// double O2_ = (O2 /(O1 + O2));
			*(tmpoutputptr++) = H / 360.1;
			*(tmpoutputptr++) = (Max - Min) / (Max + 0.1);
			*(tmpoutputptr++) = Max / 256.1;
			//*(tmpoutputptr++) = H_;
			//*(tmpoutputptr++) = H_; 
			//printf("%f %f %f\n", H_, O1_, O2_);
		}
	}
	return outputimage;
}