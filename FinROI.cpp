#include "utils.h"
#include "TransformColor.h"
#include "FitColorPDF.h"
#include "gsl\gsl_cdf.h"
#include <math.h>
#include <sstream>  
#include<iomanip>
#include<algorithm>
using namespace cv;
bool comp(ROI & r1, ROI & r2)
{
	return r1.measure < r2.measure;
}
void findroi(ROI*& outputrois, int & outputnum, Mat blurI1, Mat blurI2)
{
	Mat I1 = transformcolor(blurI1);
	Mat I2 = transformcolor(blurI2);
	Mat LabI1, LabI2;
	cvtColor(blurI1, LabI1, COLOR_BGR2Lab);
	cvtColor(blurI2, LabI2, COLOR_BGR2Lab);
	int h1 = I1.rows;
	int w1 = I1.cols;
	int size1 = h1*w1;
	double * ptr1 = (double *)I1.data;
	int binS = 10;
	int binV = 10;
	int binH = 10;
	int numbin = binH * binS + binV;
	double* hist = new double[numbin];
	memset(hist, 0, numbin * sizeof(double));
	for (int y = 0; y < h1; y++)
	{
		double * tmpptr1 = ptr1 + y * w1 * 3;
		for (int x = 0; x < w1; x++)
		{
			double H = *(tmpptr1++);
			double S = *(tmpptr1++);
			double V = *(tmpptr1++);
			if (S >= 0.1 && V >= 0.2)
			{
				int S_ = S * binS;
				int H_ = H * binH;
				int ind = H_*binS + S_;

				hist[ind]++;
			}
			else
			{
				int V_ = V * binV;
				int ind = binH*binS +  V_;
				hist[ind]++;
			}
		}
	}
	for (int ind = 0; ind < numbin; ind++)
	{
		hist[ind] /= size1;
	}

	double weight;
	double means[2];
	double covs[3];
	weight = 0;
	means[1] = means[0] = 0;
	covs[2] = covs[1] = covs[0] = 0;
	for (int y = 0; y < h1; y++)
	{
		double * tmpptr1 = ptr1 + y * w1 * 3;
		for (int x = 0; x < w1; x++)
		{
			double H = *(tmpptr1++);
			double S = *(tmpptr1++);
			double V = *(tmpptr1++);
			double p = H*S*V;
			weight += p;
			means[0] += p*x;
			means[1] += p*y;
		}
	}
	means[0] /= weight;
	means[1] /= weight;
	for (int y = 0; y < h1; y++)
	{
		double * tmpptr1 = ptr1 + y * w1 * 3;
		for (int x = 0; x < w1; x++)
		{
			double H = *(tmpptr1++);
			double S = *(tmpptr1++);
			double V = *(tmpptr1++);
			double p = H*S*V;
			double x_ = x - means[0];
			double y_ = y - means[1];
			covs[0] += (x_*x_*p / weight);
			covs[1] += (x_*y_*p / weight);
			covs[2] += (y_*y_*p / weight);
		}
	}
	double det = covs[0] * covs[2] - covs[1] * covs[1];
	double invariant = det / pow(weight, 2);
	//double det = covs[0] * covs[2] - covs[1] * covs[1];
	//a[0] = weight;
	//a[1] = pow(det, 0.25)*sqrt(weight);
	//double invariant = det / pow(weight,2);
	//int h1 = I1.rows;
	//int w1 = I1.cols;
	//int size1 = h1*w1;
	//vector<Gaussian> mixtures;
	//fitcolorpdf(mixtures, I1, h1, w1);
	//int numcolors = mixtures.size();
	//printf("nummixtures:%d\n", numcolors);
	//double* tmps = new double[numcolors];
	//unsigned char* ptr1 = I1.data;
	//double sumlogp = 0;
	////double n = 0;
	//for (int y = 0; y < h1; y++)
	//{
	//	unsigned char* rowptr = ptr1 + y * w1 * 3;
	//	for (int x = 0; x < w1; x++)
	//	{
	//		int v0 = (int)*(rowptr++);
	//		int v1 = (int)*(rowptr++);
	//		int v2 = (int)*(rowptr++);
	//		for (int i = 0; i < numcolors; i++)
	//		{
	//			double cv0 = v0 - mixtures[i].meanv0;
	//			double cv1 = v1 - mixtures[i].meanv1;
	//			double cv2 = v2 - mixtures[i].meanv2;
	//			double tv0 = mixtures[i].chol00 * cv0;
	//			double tv1 = mixtures[i].chol10 * cv0 + mixtures[i].chol11 * cv1;
	//			double tv2 = mixtures[i].chol20 * cv0 + mixtures[i].chol21 * cv1 + mixtures[i].chol22 * cv2;
	//			double sum2v = tv0*tv0 + tv1*tv1 + tv2*tv2;
	//			tmps[i] = -sum2v / 2 + mixtures[i].logw + mixtures[i].sumlogdiag;
	//		}
	//		double maxr = -DBL_MAX;
	//		for (int i = 0; i < numcolors; i++)
	//		{
	//			maxr = maxr > tmps[i] ? maxr : tmps[i];
	//		}
	//		double sum = 0;
	//		for (int i = 0; i < numcolors; i++)
	//		{
	//			sum += exp(tmps[i] - maxr);
	//		}
	//		double logsum = log(sum);
	//		//printf("%f\n", logsum + maxr);
	//		//n++;
	//		sumlogp += logsum + maxr; //sumlogp*((n-1)/n) + logsum;
	//	}
	//}
	//sumlogp /= size1;

	int h2 = I2.rows;
	int w2 = I2.cols;
	/*unsigned char* ptr2 = I2.data;
	int size2 = h2*w2;
	double* logp = new double[size2];
	for (int y = 0; y < h2; y++)
	{
		unsigned char* rowptr = ptr2 + y * w2 * 3;
		for (int x = 0; x < w2; x++)
		{
			int v0 = (int)*(rowptr++);
			int v1 = (int)*(rowptr++);
			int v2 = (int)*(rowptr++);
			for (int i = 0; i < numcolors; i++)
			{
				double cv0 = v0 - mixtures[i].meanv0;
				double cv1 = v1 - mixtures[i].meanv1;
				double cv2 = v2 - mixtures[i].meanv2;
				double tv0 = mixtures[i].chol00 * cv0;
				double tv1 = mixtures[i].chol10 * cv0 + mixtures[i].chol11 * cv1;
				double tv2 = mixtures[i].chol20 * cv0 + mixtures[i].chol21 * cv1 + mixtures[i].chol22 * cv2;
				double sum2v = tv0*tv0 + tv1*tv1 + tv2*tv2;
				tmps[i] = -sum2v / 2 + mixtures[i].logw + mixtures[i].sumlogdiag;
			}
			double maxr = -DBL_MAX;
			for (int i = 0; i < numcolors; i++)
			{
				maxr = maxr > tmps[i] ? maxr : tmps[i];
			}
			double sum = 0;
			for (int i = 0; i < numcolors; i++)
			{
				sum += exp(tmps[i] - maxr);
			}
			double logsum = log(sum);
			logp[y * w2 + x]= logsum + maxr;
		}
	}*/

	int numy = h2 / h1;
	int numx = w2 / w1;
	int numROI = numy*numx;
	//printf("%d:\n", numROI);
	ROI* rois = new ROI[4*numROI];
	int n = 0;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			numy = (h2 - h1*i*0.5) / h1;
			numx = (w2 - w1*j*0.5) / w1;
			//printf("%d,%d\n", numy, numx);
			for (int iy = 0; iy < numy; iy++)
			{
				for (int ix = 0; ix < numx; ix++)
				{
					rois[n].ly = iy*h1 + h1*i*0.5;
					rois[n].uy = rois[n].ly + h1;
					rois[n].lx = ix*w1 + +w1*j*0.5;
					rois[n].ux = rois[n].lx + w1;
					n++;
				}
			}
		}
	}
	numROI = n;

	/*int roisize = size1;
	for (int i = 0; i < numROI; i++)
	{
		int rly = rois[i].ly;
		int ruy = rois[i].uy;
		int rlx = rois[i].lx;
		int rux = rois[i].ux;
		double sumlogp_ = 0;
		for (int y = rly; y < ruy; y++)
		{
			int row = y*w2;
			double * rowptr = logp + (row + rlx);
			for (int x = rlx; x < rux; x++)
			{
				sumlogp_ += *(rowptr++);
			}
		}
		sumlogp_ /= roisize;
		rois[i].measure = (sumlogp_ - sumlogp);
		printf("%f %f %f\n", sumlogp_, sumlogp, rois[i].measure);
	}*/

	double * ptr2 = (double *) I2.data;
	int roisize = size1;
	double* hist_ = new double[numbin];
	for (int i = 0; i < numROI; i++)
	{
		int rly = rois[i].ly;
		int ruy = rois[i].uy;
		int rlx = rois[i].lx;
		int rux = rois[i].ux;
		memset(hist_, 0, numbin * sizeof(double));
		for (int y = rly; y < ruy; y++)
		{
			int row = y*w2;
			double * tmpptr2 = ptr2 + (row + rlx)*3;
			for (int x = rlx; x < rux; x++)
			{
				double H = *(tmpptr2++);
				double S = *(tmpptr2++);
				double V = *(tmpptr2++);
				if (S >= 0.1 && V >= 0.2)
				{
					int S_ = S * binS;
					int H_ = H * binH;
					int ind = H_*binS + S_;
					hist_[ind]++;
				}
				else
				{
					int V_ = V * binV;
					int ind = binH*binS + V_;
					hist_[ind]++;
				}
			}
		}
		for (int ind = 0; ind < numbin; ind++)
		{
			hist_[ind] /= roisize;
		}
		double summin = 0;
		for (int ind = 0; ind < numbin; ind++)
		{
			summin += min(hist_[ind], hist[ind]);
		}
		rois[i].measure = summin;
		//printf("%f\n", summin);
	}

	//for (int i = 0; i < numROI; i++)
	//{
	//	int rly = rois[i].ly;
	//	int ruy = rois[i].uy;
	//	int rlx = rois[i].lx;
	//	int rux = rois[i].ux;
	//	weight = 0;
	//	means[1] = means[0] = 0;
	//	covs[2] = covs[1] = covs[0] = 0;
	//	for (int y = rly; y < ruy; y++)
	//	{
	//		int row = y*w2;
	//		double * tmpptr2 = ptr2 + (row + rlx);
	//		for (int x = rlx; x < rux; x++)
	//		{
	//			double v = *(tmpptr2++);
	//			weight += v;
	//			means[0] += v*x;
	//			means[1] += v*y;
	//		}
	//	}
	//	means[0] /= weight;
	//	means[1] /= weight;
	//	for (int y = rly; y < ruy; y++)
	//	{
	//		int row = y*w2;
	//		double * tmpptr2 = ptr2 + (row + rlx);
	//		for (int x = rlx; x < rux; x++)
	//		{
	//			double v = *(tmpptr2++);
	//			double x_ = x - means[0];
	//			double y_ = y - means[1];
	//			covs[0] += (x_*x_*v / weight);
	//			covs[1] += (x_*y_*v / weight);
	//			covs[2] += (y_*y_*v / weight);
	//		}
	//	}
	//	double det = covs[0] * covs[2] - covs[1] * covs[1];
	//	double invariant_ = det / pow(weight, 2);
	//	//printf("%f %f %f\n", SSR, SST, SSR / SST);
	//	rois[i].measure = abs(invariant_ - invariant) / invariant;
	//}
	//for (int i = 0; i < numROI; i++)
	//{
	//	int rly = rois[i].ly;
	//	int ruy = rois[i].uy;
	//	int rlx = rois[i].lx;
	//	int rux = rois[i].ux;
	//	weight = 0;
	//	means[1] = means[0] = 0;
	//	covs[2] = covs[1] = covs[0] = 0;
	//	for (int y = rly; y < ruy; y++)
	//	{
	//		int row = y*w2;
	//		unsigned char* tmpptr2 = ptr2 + row + rlx;
	//		for (int x = rlx; x < rux; x++)
	//		{
	//			double v = *(tmpptr2++);
	//			weight += v;
	//			means[0] += v*x;
	//			means[1] += v*y;
	//		}
	//	}
	//	means[0] /= weight;
	//	means[1] /= weight;
	//	b[0] = weight;
	//	for (int y = rly; y < ruy; y++)
	//	{
	//		int row = y*w2;
	//		unsigned char* tmpptr2 = ptr2 + row + rlx;
	//		for (int x = rlx; x < rux; x++)
	//		{
	//			double v = *(tmpptr2++);
	//			double x_ = x - means[0];
	//			double y_ = y - means[1];
	//			covs[0] += (x_*x_*v / weight);
	//			covs[1] += (x_*y_*v / weight);
	//			covs[2] += (y_*y_*v / weight);
	//		}
	//	}
	//	double det = covs[0] * covs[2] - covs[1] * covs[1];
	//	b[1] = pow(det, 0.25)*sqrt(weight);
	//	double l1 = b[0] / a[0];
	//	double l2 = b[1] / a[1];
	//	double p = l1 / l2;
	//	//double F = (SSR) / (SSE/4);
	//	printf("%f\n", SSE/roisize);
	//	rois[i].measure = SSE/roisize;
	//}

	double tol = 0.6;
	n = 0;
	//printf("%f %f %f %f %f\n", min0, min1, min2, min, tol);
	for (int i = 0; i < numROI; i++)
	{
		//printf("%f %f\n", rois[i].measure[0], tol);
		//printf("%f %f %f %f %f %f\n",min0,min1,min2,min, tol, rois[i].measure);
		if (rois[i].measure > tol)//&& rois[i].measure[1] < tol && rois[i].measure[2] < tol)
		{
			rois[n] = rois[i];
			n++;
		}
	}
	numROI = n;

	

	double disscales[2] = { 0.15,0.075};
	int round = 0;
	int maxround = 2;
	double b[3];
	while (round < maxround)
	{
		double disscale = disscales[round];
		round++;
		for (int i = 0; i < numROI; i++)
		{
			//printf("%d %d %d %d _", rois[i].ly, rois[i].uy, rois[i].lx, rois[i].ux);
			double radiusy = double(h1) / 2;
			double radiusx = double(w1) / 2;
			double displaycey = h1 * disscale;
			double displaycex = w1 * disscale;
			//printf("%f\n", disscale);
			double cy = double(rois[i].ly + rois[i].uy) / 2;
			double cx = double(rois[i].lx + rois[i].ux) / 2;
			double minmeasure = DBL_MAX; 0;
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					for (int l = -1; l <= 1; l++)
					{
						for (int m = -1; m <= 1; m++)
						{
							int rly = cy + j*displaycey - (disscale * l  + 1) * radiusy;
							int ruy = cy + j*displaycey + (disscale * l  + 1) * radiusy;
							int rlx = cx + k*displaycex - (disscale * m  + 1) * radiusx;
							int rux = cx + k*displaycex + (disscale * m  + 1) * radiusx;
							if (rly < 0 || ruy >= h2 || rlx < 0 || rux >= w2)
							{
								continue;
							}
							weight = 0;
							means[1] = means[0] = 0;
							covs[2] = covs[1] = covs[0] = 0;
							for (int y = rly; y < ruy; y++)
							{
								int row = y*w2;
								double* tmpptr2 = ptr2 + (row + rlx) * 3;
								for (int x = rlx; x < rux; x++)
								{
									double H = *(tmpptr2++);
									double S = *(tmpptr2++);
									double V = *(tmpptr2++);
									double p = H*S*V;
									weight += p;
									means[0] += p*x;
									means[1] += p*y;
								}
							}
							means[0] /= weight;
							means[1] /= weight;
							for (int y = rly; y < ruy; y++)
							{
								int row = y*w2;
								double * tmpptr2 = ptr2 + (row + rlx) * 3;
								for (int x = rlx; x < rux; x++)
								{
									double H = *(tmpptr2++);
									double S = *(tmpptr2++);
									double V = *(tmpptr2++);
									double p = H*S*V;
									double x_ = x - means[0];
									double y_ = y - means[1];
									covs[0] += (x_*x_*p / weight);
									covs[1] += (x_*y_*p / weight);
									covs[2] += (y_*y_*p / weight);
								}
							}	
							double det = covs[0] * covs[2] - covs[1] * covs[1];
							double invariant_ = det / pow(weight, 2);
							double measure = abs(invariant - invariant_);
							//printf("%f\n", measure);
							//printf("%f %f %f %f %f %f %f\n", b[0],b[1],b[2],b[3],a[4],a[5],measure);
							/*double det = covs[0] * covs[2] - covs[1] * covs[1];
							double invariant_ = det / pow(weight, 2);
							double measure = abs(invariant - invariant_)/ invariant;*/
							//b[0] = weight;
							//b[1] = pow(det, 0.25)*sqrt(weight);
							//double k0 = b[0] / a[0];
							//double k1 = b[1] / a[1];
							//double k = (b[0] * a[0] + b[1] * a[1]) / (a[0]*a[0]+a[1]*a[1]);
							//double SST = pow(b[1] - b[0], 2) / 2;
							//double SSR = pow(a[0] * k - b[0], 2) + pow(a[1] * k - b[1], 2);
							//measure = (k1*k0 + 1) / (sqrt(k1*k1 + 1)*sqrt(k0*k0 + 1));
							//measure = SSR / SST;
							//printf("%f %f %f %f\n", invariant, invariant_, abs(invariant - invariant_), abs(invariant - invariant_) / invariant);
							//double F = (SSR) / (SSE/4);
							if (measure < minmeasure)
							{
								minmeasure = measure;
								rois[i].measure = measure;
								rois[i].ly = rly;
								rois[i].uy = ruy;
								rois[i].lx = rlx;
								rois[i].ux = rux;
							}
						}
					}
				}
			}
			//printf("%d %d %d %d\n", rois[i].lx, rois[i].ux, rois[i].ly, rois[i].uy);
		}
		//printf("\n");
	}

	for (int i = 0; i < numROI; i++)
	{
		int rly = rois[i].ly;
		int ruy = rois[i].uy;
		int rlx = rois[i].lx;
		int rux = rois[i].ux;
		memset(hist_, 0, numbin * sizeof(double));
		for (int y = rly; y < ruy; y++)
		{
			int row = y*w2;
			double * tmpptr2 = ptr2 + (row + rlx) * 3;
			for (int x = rlx; x < rux; x++)
			{
				double H = *(tmpptr2++);
				double S = *(tmpptr2++);
				double V = *(tmpptr2++);
				if (S >= 0.1 && V >= 0.2)
				{
					int S_ = S * binS;
					int H_ = H * binH;
					int ind = H_*binS + S_;
					hist_[ind]++;
				}
				else
				{
					int V_ = V * binV;
					int ind = binH*binS + V_;
					hist_[ind]++;
				}
			}
		}
		for (int ind = 0; ind < numbin; ind++)
		{
			hist_[ind] /= roisize;
		}
		double summin = 0;
		for (int ind = 0; ind < numbin; ind++)
		{
			summin += min(hist_[ind], hist[ind]);
		}
		rois[i].measure = 1 - summin;
		//printf("%f\n", summin);
	}
	printf("%d:\n", numROI);

	bool* mask = new bool[numROI];
	memset(mask, false, sizeof(bool)*numROI);
	for (int i = 0; i < numROI; i++)
	{
		if (mask[i])
		{
			continue;
		}
		int rly = rois[i].ly;
		int ruy = rois[i].uy;
		int rlx = rois[i].lx;
		int rux = rois[i].ux;
		int leny = ruy - rly;
		int lenx = rux - rlx;
		int size = leny*lenx;
		for (int j = 0; j < numROI; j++)
		{
			if (j == i)
			{
				continue;
			}
			int tly = rois[j].ly;
			int tuy = rois[j].uy;
			int tleny = tuy - tly;
			int ruy_ = ruy > tuy ? ruy : tuy;
			int rly_ = rly < tly ? rly : tly;
			int leny_ = ruy_ - rly_;
			if (leny_ > leny + tleny)
			{
				continue;
			}
			int tlx = rois[j].lx;
			int tux = rois[j].ux;
			int tlenx = tux - tlx;
			int rux_ = rux > tux ? rux : tux;
			int rlx_ = rlx < tlx ? rlx : tlx;
			int lenx_ = rux_ - rlx_;
			if (lenx_ > lenx + tlenx)
			{
				continue;
			}
			int tsize = tleny*tlenx;
			int overlapx = lenx + tlenx - lenx_;
			int overlapy = leny + tleny - leny_;
			int overlapsize = overlapx * overlapy;
			//int unionsize = lenx*leny+tlenx*tleny - overlapsize;
			//int unionsize = size+tsize - overlapsize;
			double ratio1 = (double)overlapsize / size;
			double ratio2 = (double)overlapsize / tsize;
			//printf("%f\n", ratio);
			if (ratio1 > 0.6 || ratio2 > 0.6)
			{
				
				if (rois[i].measure <= rois[j].measure)
				{
					mask[j] = true;
				}
				else
				{
					mask[i] = true;
					break;
				}
			}
		}
	}
	n = 0;
	for (int i = 0; i < numROI; i++)
	{
		if (!mask[i])
		{
			//printf("%f\n", rois[i].measure);
			rois[n] = rois[i];
			n++;
		}
	}
	numROI = n;

	printf("numROI:%d", numROI);
	Mat tmpI2 = blurI2.clone();
	Point** curves = new Point*[numROI];
	int * npts = new int[numROI];
	std::stringstream ss;
	for (int i = 0; i < numROI; i++)
	{
		Point* curve = new Point[4];
		curve[0] = Point(rois[i].lx, rois[i].ly);
		curve[1] = Point(rois[i].ux, rois[i].ly);
		curve[2] = Point(rois[i].ux, rois[i].uy);
		curve[3] = Point(rois[i].lx, rois[i].uy);
		curves[i] = curve;
		npts[i] = 4;
		ss << rois[i].measure;
		string t;
		ss >> std::setprecision(4) >> t;
		ss.clear();
		putText(tmpI2, t, curve[0], CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0));
	}
	bool iscurveclosed = 1;
	int thickline = 1;
	polylines(tmpI2, (const Point **)curves, npts, numROI, iscurveclosed, Scalar(0, 255, 255), 2);
	namedWindow("i2", CV_WINDOW_NORMAL);
	imshow("i2", tmpI2);
	waitKey(0);

	
	outputnum = numROI;
	outputrois = rois;

	unsigned char* labptr1 = LabI1.data;
	unsigned char* labptr2 = LabI2.data;
	Config* configs = new Config[numROI];
	AffineMat* mats = new AffineMat[numROI];
	for (int i = 0; i < numROI; i++)
	{
		double h = double(rois[i].uy - rois[i].ly);
		double w = double(rois[i].ux - rois[i].lx);
		configs[i].sx = w / w1;
		configs[i].sy = h / h1;
		configs[i].r1 = 0;
		configs[i].r2 = 0;
		configs[i].tx = double(rois[i].lx + rois[i].ux) / 2;
		configs[i].ty = double(rois[i].ly + rois[i].uy) / 2;
		//printf("%d %d %d %d\n", rois[i].lx, rois[i].ux, rois[i].ly, rois[i].uy);
		//printf("%f %f %f %f\n", configs[i].sx, configs[i].sy, configs[i].tx, configs[i].ty);
	}
	double scale = 0.1;
	double rotation1 = 3.14 / 24;
	double rotation2 = 3.14 / 24;
	double translationy = double(h1) / 4;
	double translationx = double(w1) / 4;  
	double ratio = 0.9;
	maxround = 6;  
	double* dists = new double[numROI];
	for (int round = 0; round < maxround; round++)
	{
		int mode = round % 3;
		int ranget = 0;
		int ranges = 0;
		int ranger = 0;
		if (mode == 0)
		{
			ranget = 1;
			ranger = 1;
			//ranges
		}
		else if(mode == 1)
		{
			ranger = 1;
			ranges = 1;
		}
		else
		{
			ranget = 1;
			ranges = 1;
		}
		double min = DBL_MAX;
		for (int i = 0; i < numROI; i++)
		{
			double mindist = DBL_MAX;
			Config minconfig;
			AffineMat minmat;
			for (int ity = -ranget; ity <= ranget; ity++)
			{
				double ty = configs[i].ty + configs[i].sy*translationy*ity;
				for (int itx = -ranget; itx <= ranget; itx++)
				{
					double tx = configs[i].tx + configs[i].sx*translationx*itx;
					for (int isy = -ranges; isy <= ranges; isy++)
					{
						double sy = configs[i].sy + scale*isy;
						for (int isx = -ranges; isx <= ranges; isx++)
						{
							double sx = configs[i].sx  + scale*isx;
							for (int ir1 = -ranger; ir1 <= ranger; ir1++)
							{
								double r1 = configs[i].r1 + ir1* rotation1;
								for (int ir2 = -ranger; ir2 <= ranger; ir2++)
								{
									double r2 = configs[i].r2 + ir2* rotation2;
									double a00 = sx*cos(r1)*cos(r2) - sy*sin(r1)*sin(r2);
									double a01 = -sx*cos(r1)*sin(r2) - sy*cos(r2)*sin(r1);
									double a02 = tx;
									double a10 = sx*cos(r2)*sin(r1) + sy*cos(r1)*sin(r2);
									double a11 = sy*cos(r1)*cos(r2) - sx*sin(r1)*sin(r2);
									double a12 = ty;
									int r1x = -w1 / 2;
									int r1y = -h1 / 2;
									int c1x = a00 * (-r1x) + a01 * (-r1y) + a02;
									int c1y = a10 * (-r1x) + a11 * (-r1y) + a12;
									int c2x = a00 * (r1x)+a01 * (-r1y) + a02;
									int c2y = a10 * (r1x)+a11 * (-r1y) + a12;
									int c3x = a00 * (r1x)+a01 * (r1y)+a02;
									int c3y = a10 * (r1x)+a11 * (r1y)+a12;
									int c4x = a00 * (-r1x) + a01 * (r1y)+a02;
									int c4y = a10 * (-r1x) + a11 * (r1y)+a12;
									//printf("%d: sx:%f tx:%f sy:%f ty:%f\n", i, a00, c1x + a02, a11, c1y + a12);
									bool inside = ((c1x>=0) && (c1x<w2) && (c1y>=0) && (c1y<h2) &&
										(c2x>=0) && (c2x<w2) && (c2y>=0) && (c2y<h2 ) &&
										(c3x>=0) && (c3x<w2) && (c3y>=0) && (c3y<h2 ) &&
										(c4x>=0) && (c4x<w2) && (c4y>=0) && (c4y<h2));
									if (!inside)
									{
										printf("???\n");
										continue;
									}
									double dist = 0;
									double size1 = 0;
									for (int y1 = 0; y1 < h1; y1++)
									{
										int row1 = y1*w1;
										int ry1 = y1 - h1 / 2;
										for (int x1 = 0; x1 < w1; x1++)
										{
											int rx1 = x1 - w1 / 2;
											int ind1 = row1 + x1;
											ind1 *= 3;
											int x2 = a00*rx1 + a01*ry1 + a02;
											int y2 = a10*rx1 + a11*ry1 + a12;
											int ind2 = y2*w2 + x2;
											ind2 *= 3;
											int d0 = int(labptr2[ind2]) - labptr1[ind1];
											int d1 = int(labptr2[ind2 + 1]) - labptr1[ind1 + 1];
											int d2 = int(labptr2[ind2 + 2]) - labptr1[ind1 + 2];
											double sum = d0*d0 + d1*d1 + d2*d2;
											size1++;
											dist = sum / size1 + dist*((size1 - 1) / size1);
										}
									}
									if (dist < mindist)
									{
										mindist = dist;
										minconfig.sx = sx;
										minconfig.sy = sy;
										minconfig.tx = tx;
										minconfig.ty = ty;
										minconfig.r1 = r1;
										minconfig.r2 = r2;
										minmat.a00 = a00;
										minmat.a01 = a01;
										minmat.a02 = a02;
										minmat.a10 = a10;
										minmat.a11 = a11;
										minmat.a12 = a12;
									}
								}
							}
						}
					}
				}
			}
			configs[i] = minconfig;
			mats[i] = minmat;
			dists[i] = mindist;
			min = min < mindist ? min : mindist;
			//printf("%d %f\n",i, dists[i]);
		}
		if (mode == 0)
		{
			translationy *= ratio;
			translationx *= ratio;
			rotation2 *= ratio;
			rotation1 *= ratio;
		}
		else if(mode == 1)
		{
			rotation2 *= ratio;
			rotation1 *= ratio;
			scale *= ratio;
		}
		else
		{
			translationy *= ratio;
			translationx *= ratio;
			scale *= ratio;
		}
		//if(round < 6)
		//{ 
		//	scale *= ratio;
		//	rotation2 *= ratio;
		//	rotation1 *= ratio;
		//	translationy *= ratio;
		//	translationx *= ratio;
		//}
		//else if(round == 6)
		//{
		//	scale = 0.08;
		//	rotation1 = 3.14 / 24;
		//	rotation2 = 3.14 / 12;
		//	translationy = double(h1) / 8;
		//	translationx = double(w1) / 8;
		//	min = DBL_MAX;
		//}
		//else
		//{
		//	min = DBL_MAX;
		//}
		//n = 0;
		////printf("%f %f %f %f %f\n", min0, min1, min2, min, tol);
		//for (int i = 0; i < numROI; i++)
		//{
		//	printf("%f %f %f\n", dists[i], min, min + tols[round]);
		//	//printf("%f %f %f %f %f %f\n",min0,min1,min2,min, tol, rois[i].measure);
		//	if (dists[i] < (min + tols[round]))//&& rois[i].measure[1] < tol && rois[i].measure[2] < tol)
		//	{
		//		configs[n] = configs[i];
		//		mats[n] = mats[i];
		//		dists[n] = dists[i];
		//		n++;
		//	}
		//}
		//numROI = n;
		//printf("round:%d numROI:%d %f\n", round, numROI, min + tols[round]*w1*h1);
		//if (round == maxround - 1)
		//{
		//	double* tmpdists = new double[numROI];
		//	memcpy(tmpdists, dists, numROI * sizeof(double));
		//	std::sort(tmpdists, tmpdists + numROI);
		//	double s = tmpdists[1] - tmpdists[0];
		//	if (s < 1)
		//	{
		//		s = tmpdists[2] - tmpdists[0];
		//	}
		//	double min = tmpdists[0];
		//	s = s*1.8 < 200 ? s * 1.8 : 200;
		//	n = 0;
		//	for (int i = 0; i < numROI; i++)
		//	{
		//		printf("%f %f %f\n", dists[i], min, min + s);
		//		//printf("%f %f %f %f %f %f\n",min0,min1,min2,min, tol, rois[i].measure);
		//		if (dists[i] < (min + s))//&& rois[i].measure[1] < tol && rois[i].measure[2] < tol)
		//		{
		//			configs[n] = configs[i];
		//			mats[n] = mats[i];
		//			dists[n] = dists[i];
		//			n++;
		//		}
		//	}
		//	numROI = n;
		//	printf("%d\n", numROI);
		//}
		Point** curves = new cv::Point*[numROI];
		int* npts = new int[numROI];
		std::stringstream ss;
		Mat tmp2 = blurI2.clone();
		for (int i = 0; i < numROI; i++)
		{
			cv::Point* curve = new cv::Point[4];
			AffineMat mat = mats[i];
			int r1x = -w1 / 2;
			int r1y = -h1 / 2;
			int c1x = mat.a00 * (-r1x) + mat.a01 * (-r1y) + mat.a02;
			int c1y = mat.a10 * (-r1x) + mat.a11 * (-r1y) + mat.a12;
			int c2x = mat.a00 * (r1x)+mat.a01 * (-r1y) + mat.a02;
			int c2y = mat.a10 * (r1x)+mat.a11 * (-r1y) + mat.a12;
			int c3x = mat.a00 * (r1x)+mat.a01 * (r1y)+mat.a02;
			int c3y = mat.a10 * (r1x)+mat.a11 * (r1y)+mat.a12;
			int c4x = mat.a00 * (-r1x) + mat.a01 * (r1y)+mat.a02;
			int c4y = mat.a10 * (-r1x) + mat.a11 * (r1y)+mat.a12;
			printf("%f\n", dists[i]);
			//printf("%f %f %f %f %f %f\n",mat.a00,mat.a01,mat.a02,mat.a10,mat.a11,mat.a12);
			//printf("%d %d %d %d %d %d %d %d\n", c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y);
			curve[0] = cv::Point(c1x, c1y);
			curve[1] = cv::Point(c2x, c2y);
			curve[2] = cv::Point(c3x, c3y);
			curve[3] = cv::Point(c4x, c4y);

			curves[i] = curve;
			npts[i] = 4;
			ss << dists[i];
			string t;
			ss >> std::setprecision(4) >> t;
			ss.clear();
			putText(tmp2, t, curve[2], CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0));
		}
		bool iscurveclosed = 1;
		bool thickline = 1;
		polylines(tmp2, (const Point **)curves, npts, numROI, iscurveclosed, Scalar(0, 255, 255), 2);
		namedWindow("i2", CV_WINDOW_NORMAL);
		imshow("i2", tmp2);
		printf("??\n");
		waitKey(0);
	}
	for (int i = 0; i < numROI; i++)
	{
		cv::Point* curve = new cv::Point[4];
		AffineMat mat = mats[i];
		int r1x = -w1 / 2;
		int r1y = -h1 / 2;
		int c1x = mat.a00 * (-r1x) + mat.a01 * (-r1y) + mat.a02;
		int c1y = mat.a10 * (-r1x) + mat.a11 * (-r1y) + mat.a12;
		int c2x = mat.a00 * (r1x)+mat.a01 * (-r1y) + mat.a02;
		int c2y = mat.a10 * (r1x)+mat.a11 * (-r1y) + mat.a12;
		int c3x = mat.a00 * (r1x)+mat.a01 * (r1y)+mat.a02;
		int c3y = mat.a10 * (r1x)+mat.a11 * (r1y)+mat.a12;
		int c4x = mat.a00 * (-r1x) + mat.a01 * (r1y)+mat.a02;
		int c4y = mat.a10 * (-r1x) + mat.a11 * (r1y)+mat.a12;
		//printf("%f %f %f %f %f %f\n",mat.a00,mat.a01,mat.a02,mat.a10,mat.a11,mat.a12);
		//printf("%d %d %d %d %d %d %d %d\n", c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y);
		int rlx = w2;
		int rux = 0;
		int rly = h2;
		int ruy = 0;
		rlx = rlx < c1x ? rlx : c1x;
		rlx = rlx < c2x ? rlx : c2x;
		rlx = rlx < c3x ? rlx : c3x;
		rlx = rlx < c4x ? rlx : c4x;
		rux = rux > c1x ? rux : c1x;
		rux = rux > c2x ? rux : c2x;
		rux = rux > c3x ? rux : c3x;
		rux = rux > c4x ? rux : c4x;
		rly = rly < c1y ? rly : c1y;
		rly = rly < c2y ? rly : c2y;
		rly = rly < c3y ? rly : c3y;
		rly = rly < c4y ? rly : c4y;
		ruy = ruy > c1y ? ruy : c1y;
		ruy = ruy > c2y ? ruy : c2y;
		ruy = ruy > c3y ? ruy : c3y;
		ruy = ruy > c4y ? ruy : c4y;
		rois[i].ly = rly;
		rois[i].uy = ruy;
		rois[i].lx = rlx;
		rois[i].ux = rux;
		rois[i].measure = dists[i];
	}

	memset(mask, false, sizeof(bool)*numROI);
	for (int i = 0; i < numROI; i++)
	{
		if (mask[i])
		{
			continue;
		}
		int rly = rois[i].ly;
		int ruy = rois[i].uy;
		int rlx = rois[i].lx;
		int rux = rois[i].ux;
		int leny = ruy - rly;
		int lenx = rux - rlx;
		int size = leny*lenx;
		for (int j = 0; j < numROI; j++)
		{
			if (j == i)
			{
				continue;
			}
			int tly = rois[j].ly;
			int tuy = rois[j].uy;
			int tleny = tuy - tly;
			int ruy_ = ruy > tuy ? ruy : tuy;
			int rly_ = rly < tly ? rly : tly;
			int leny_ = ruy_ - rly_;
			if (leny_ > leny + tleny)
			{
				continue;
			}
			int tlx = rois[j].lx;
			int tux = rois[j].ux;
			int tlenx = tux - tlx;
			int rux_ = rux > tux ? rux : tux;
			int rlx_ = rlx < tlx ? rlx : tlx;
			int lenx_ = rux_ - rlx_;
			if (lenx_ > lenx + tlenx)
			{
				continue;
			}
			int tsize = tleny*tlenx;
			int overlapx = lenx + tlenx - lenx_;
			int overlapy = leny + tleny - leny_;
			int overlapsize = overlapx * overlapy;
			//int unionsize = lenx*leny+tlenx*tleny - overlapsize;
			//int unionsize = size+tsize - overlapsize;
			double ratio1 = (double)overlapsize / size;
			double ratio2 = (double)overlapsize / tsize;
			//printf("%f\n", ratio);
			if (ratio1 > 0.4 || ratio2 > 0.4)
			{

				if (rois[i].measure <= rois[j].measure)
				{
					mask[j] = true;
				}
				else
				{
					mask[i] = true;
					break;
				}
			}
		}
	}
	n = 0;
	double min = DBL_MAX;
	for (int i = 0; i < numROI; i++)
	{
		if (!mask[i])
		{
			//printf("%f\n", rois[i].measure);
			rois[n] = rois[i];
			n++;
		}
	}
	numROI = n;
	std::sort(rois, rois + numROI, comp);
	double range = rois[1].measure - rois[0].measure;
	min = rois[0].measure;
	range = range*1.3 < 200 ? range * 1.3 : 200;
	//range = range > 60 ? range : 60;
	n = 0;
	for (int i = 0; i < numROI; i++)
	{
		printf("%f %f\n", rois[i].measure, min + range);
		if (rois[i].measure < (min + range))//&& rois[i].measure[1] < tol && rois[i].measure[2] < tol)
		{
			rois[n] = rois[i];
			n++;
		}
	}

	numROI = n;
	tmpI2 = blurI2.clone();
	for (int i = 0; i < numROI; i++)
	{
		Point* curve = new Point[4];
		curve[0] = Point(rois[i].lx, rois[i].ly);
		curve[1] = Point(rois[i].ux, rois[i].ly);
		curve[2] = Point(rois[i].ux, rois[i].uy);
		curve[3] = Point(rois[i].lx, rois[i].uy);
		curves[i] = curve;
		npts[i] = 4;
		ss << rois[i].measure;
		string t;
		ss >> std::setprecision(4) >> t;
		ss.clear();
		putText(tmpI2, t, curve[0], CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0));
	}
	iscurveclosed = 1;
	thickline = 1;
	polylines(tmpI2, (const Point **)curves, npts, numROI, iscurveclosed, Scalar(0, 255, 255), 2);
	namedWindow("i2", CV_WINDOW_NORMAL);
	imshow("i2", tmpI2);
	waitKey(0);
}