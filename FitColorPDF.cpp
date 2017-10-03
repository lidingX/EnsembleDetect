#include "utils.h"
#include "GMMFitting.h"
#include "opencv2/imgproc/imgproc.hpp"  
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <math.h>
using namespace cv;
using namespace std;
struct Hist
{
	int w;
	int histnum;
};
struct Coord
{
	double totalvar;
	int histid;
	int ind;
	int w;
};
struct Colorcluster
{
	double meanv0;
	double meanv1;
	double meanv2;
	double w;
};
bool comp(Coord & m1, Coord & m2)
{
	return m1.w == m2.w ? m1.totalvar < m2.totalvar : m1.w > m2.w;
};
void fitcolorpdf(vector<Gaussian>& gmm, Mat I1, int h1, int w1)
{
	//L 0-255, a 1-255, b 1-255
	int numv0 = 16;
	int numv1 = 16;
	int numv2 = 16;
	int v0size = 256 / numv0;
	int v1size = 256 / numv1;
	int v2size = 256 / numv2;
	int loopv2 = 1;
	int loopv1 = numv2 * loopv2;
	int loopv0 = numv1 * loopv1;
	int loop = numv0 * loopv0;
	int* hist = new int[loop];
	memset(hist, 0, loop * sizeof(int));
	int imsize = h1*w1;
	Coord* points = new Coord[imsize];
	double* vars = new double[imsize];
	double* imptr = new double[imsize * 3];
	double* tmpimptr = imptr;
	unsigned char* ucharptr = I1.data;
	for (int i = 0; i < imsize * 3; i++)
	{
		*(tmpimptr++) = (double)*(ucharptr++);
	}
	int pind = 0;
	int cutting = 5;
	int smoothr = 1;
	int numpixels = (h1 - 2 * cutting)*(w1 - 2 * cutting);
	int n = (2 * smoothr + 1)*(2 * smoothr + 1);
	for (int cy = cutting; cy < h1 - cutting; cy++)
	{
		for (int cx = cutting; cx < w1 - cutting; cx++)
		{
			int cind = cy*w1 + cx;
			double cv0 = imptr[cind * 3];
			double cv1 = imptr[cind * 3 + 1];
			double cv2 = imptr[cind * 3 + 2];
			int icv0 = cv0 / v0size;
			int icv1 = cv1 / v1size;
			int icv2 = cv2 / v2size;
			int histid = icv0 * loopv0 + icv1*loopv1 + icv2;
			hist[histid] += 1;
			double var = DBL_MAX;
			for (int y = cy - smoothr; y <= cy + smoothr; y++)
			{
				for (int x = cx - smoothr; x <= cx + smoothr; x++)
				{
					if (y == cy && x == cx)
					{
						continue;
					}
					int ind = y*w1 + x;
					double v0 = imptr[ind * 3];
					double v1 = imptr[ind * 3 + 1];
					double v2 = imptr[ind * 3 + 2];
					double dist = pow(v0 - cv0,2) + pow(v1 - cv1,2) + pow(v2 - cv2,2);
					var = var < dist ? var : dist;
				}
			}
			vars[cind] = var;
			points[pind].totalvar = var;
			points[pind].histid = histid;
			points[pind].ind = cind;
			pind++;
		}
	}
	int num_point = pind;
	//backpropagation
	for (int i = 0; i < num_point; i++)
	{
		points[i].w = hist[points[i].histid];
	}
	sort(points, points + num_point, comp);
	cv::Mat showmat(h1, w1, CV_8UC3);
	int round = 0;
	int maxround = 1;
	double alpha = 700;
	int numcolors;
	vector<Colorcluster> finalclusters;
	bool * clusteredmap = new bool[imsize];
	while (round < maxround)
	{
		round++;
		//alpha += alphaincrement;
		memset(clusteredmap, -1, sizeof(bool)*imsize);
		for (int y = cutting; y < h1 - cutting; y++)
		{
			for (int x = cutting; x < w1 - cutting; x++)
			{
				int ind = y*w1 + x;
				clusteredmap[ind] = false;
			}
		}
		vector<Colorcluster> clusters;
		numcolors = 0;
		for (int i = 0; i < num_point; i++)
		{
			int ind = points[i].ind;
			if (clusteredmap[ind])
			{
				continue;
			}
			clusteredmap[ind] = true;
			double meanv0;
			double meanv1;
			double meanv2;
			double n;
			double v0 = imptr[ind * 3];
			double v1 = imptr[ind * 3 + 1];
			double v2 = imptr[ind * 3 + 2];
			bool add = false;
			int colorind;
			for (int j = 0; j < numcolors; j++)
			{
				meanv0 = clusters[j].meanv0;
				meanv1 = clusters[j].meanv1;
				meanv2 = clusters[j].meanv2;
				double distv0 = pow(v0 - meanv0,2);
				double distv1 = pow(v1 - meanv1,2);
				double distv2 = pow(v2 - meanv2,2);
				if (distv0 + distv1 + distv2 <= alpha)
				{
					n = clusters[j].w;
					n++;
					double oldmeanv0 = meanv0;
					double oldmeanv1 = meanv1;
					double oldmeanv2 = meanv2;
					meanv0 = oldmeanv0 * ((n - 1) / n) + v0 / n;
					meanv1 = oldmeanv1 * ((n - 1) / n) + v1 / n;
					meanv2 = oldmeanv2 * ((n - 1) / n) + v2 / n;
					add = true;
					colorind = j;
					break;
				}
			}
			if (!add)
			{
				meanv0 = v0;
				meanv1 = v1;
				meanv2 = v2;
				n = 1;
			}
			queue<int> bfs;
			bfs.push(ind);
			while (!bfs.empty())
			{
				int cind = bfs.front();
				bfs.pop();
				int cy = cind / w1;
				int cx = cind % w1;
				for (int y = cy - 1; y <= cy + 1; y++)
				{
					for (int x = cx - 1; x <= cx + 1; x++)
					{
						int ind = y*w1 + x;
						if (!clusteredmap[ind])
						{
							double v0 = imptr[ind * 3];
							double v1 = imptr[ind * 3 + 1];
							double v2 = imptr[ind * 3 + 2];
							double distv0 = pow(v0 - meanv0,2);
							double distv1 = pow(v1 - meanv1,2);
							double distv2 = pow(v2 - meanv2,2);
							if (distv0 + distv1 + distv2 <= alpha)
							{
								n++;
								double oldmeanv0 = meanv0;
								double oldmeanv1 = meanv1;
								double oldmeanv2 = meanv2;
								meanv0 = oldmeanv0 * ((n - 1) / n) + v0 / n;
								meanv1 = oldmeanv1 * ((n - 1) / n) + v1 / n;
								meanv2 = oldmeanv2 * ((n - 1) / n) + v2 / n;
								//printf("%f %f %f %f %f %f\n", varv0, varv1, varv2,var[ind],var[ind+1],var[ind+2]);
								clusteredmap[ind] = true;
								if (vars[ind]  < alpha)//beta*(varv0 + varv1 + varv2))
								{
									bfs.push(ind);
								}
							}
						}
					}
				}
			}
			if (!add)
			{
				Colorcluster color;
				color.meanv0 = meanv0;
				color.meanv1 = meanv1;
				color.meanv2 = meanv2;
				color.w = n;
				clusters.push_back(color);
				numcolors++;
			}
			else
			{
				clusters[colorind].meanv0 = meanv0;
				clusters[colorind].meanv1 = meanv1;
				clusters[colorind].meanv2 = meanv2;
				clusters[colorind].w = n;
			}
		}
		printf("%d %d\n", round, numcolors);
		if (round == maxround)
		{
			finalclusters = clusters;
			for (int y = cutting; y < h1 - cutting; y++)
			{
				for (int x = cutting; x < w1 - cutting; x++)
				{
					double v0 = imptr[(y*w1 + x) * 3];
					double v1 = imptr[(y*w1 + x) * 3 + 1];
					double v2 = imptr[(y*w1 + x) * 3 + 2];
					double mindist = DBL_MAX;
					int minid = 0;
					for (int i = 0; i < numcolors; i++)
					{
						double dist = pow(v0 - finalclusters[i].meanv0,2) + pow(v1 - finalclusters[i].meanv1,2) + pow(v2 - finalclusters[i].meanv2,2);
						if (dist < mindist)
						{
							mindist = dist;
							minid = i;
						}
					}
					imptr[(y*w1 + x) * 3] = finalclusters[minid].meanv0;
					imptr[(y*w1 + x) * 3 + 1] = finalclusters[minid].meanv1;
					imptr[(y*w1 + x) * 3 + 2] = finalclusters[minid].meanv2;
				}
			}
			unsigned char* matptr = showmat.data;
			for (int y = 0; y < h1; y++)
			{
				for (int x = 0; x < w1; x++)
				{
					matptr[(y*w1 + x) * 3] = imptr[(y*w1 + x) * 3];
					matptr[(y*w1 + x) * 3 + 1] = imptr[(y*w1 + x) * 3 + 1];
					matptr[(y*w1 + x) * 3 + 2] = imptr[(y*w1 + x) * 3 + 2];
				}
			}
			cv::cvtColor(showmat, showmat, cv::COLOR_Lab2BGR);
			//cv::cvtColor(showmat, showmat, cv::COLOR_HSV2BGR);
			cv::namedWindow("show", CV_WINDOW_NORMAL);
			cv::imshow("show", showmat);
			cv::waitKey(0);
		}
	}
	int* labels = new int[imsize];
	int* lptr = labels;
	ucharptr = I1.data;
	for (int y = 0; y < h1; y++)
	{
		for (int x = 0; x < w1; x++)
		{
			double v0 = ucharptr[(y*w1 + x) * 3];
			double v1 = ucharptr[(y*w1 + x) * 3 + 1];
			double v2 = ucharptr[(y*w1 + x) * 3 + 2];
			double mindist = DBL_MAX;
			int minid = 0;
			for (int i = 0; i < numcolors; i++)
			{
				double dist = pow(v0 - finalclusters[i].meanv0,2) + pow(v1 - finalclusters[i].meanv1,2) + pow(v2 - finalclusters[i].meanv2,2);
				if (dist < mindist)
				{
					mindist = dist;
					minid = i;
				}
			}
			*(lptr++) = minid;
		}
	}
	int n_components = numcolors;
	double* weights_;
	double* logweights_;
	double* sumlogdiags_;
	double* means_;
	double* precisions_cholesky_;
	int maxiter = 50;
	double tol = 1e-2;
	gmmfitting(n_components, weights_, logweights_, sumlogdiags_, means_, precisions_cholesky_, imptr, labels, numpixels, 3, n_components, tol, maxiter);
	for (int i = 0; i < n_components; i++)
	{
		printf("%lf\n", weights_[i]);
		Gaussian gaussian;
		gaussian.weight = weights_[i];
		gaussian.meanv0 = means_[3 * i];
		gaussian.meanv1 = means_[3 * i + 1];
		gaussian.meanv2 = means_[3 * i + 2];
		gaussian.chol00 = precisions_cholesky_[9 * i];
		gaussian.chol10 = precisions_cholesky_[9 * i + 3];
		gaussian.chol11 = precisions_cholesky_[9 * i + 3 + 1];
		gaussian.chol20 = precisions_cholesky_[9 * i + 6];
		gaussian.chol21 = precisions_cholesky_[9 * i + 6 + 1];
		gaussian.chol22 = precisions_cholesky_[9 * i + 6 + 2];
		gaussian.sumlogdiag = sumlogdiags_[i];
		gaussian.logw = logweights_[i];
		gmm.push_back(gaussian);
	}
}

