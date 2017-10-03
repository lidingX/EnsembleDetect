#pragma once
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/core/core.hpp"  
#include "opencv2/legacy/legacy.hpp" 
struct ROI
{
	int lx;
	int ux;
	int ly;
	int uy;
	double measure;
};
struct Config
{
	double tx;
	double ty;
	double sx;
	double sy;
	double r2;
	double r1;
};
struct AffineMat
{
	double a00;
	double a01;
	double a02;
	double a10;
	double a11;
	double a12;
};
struct Gaussian
{
	double meanv0;
	double meanv1;
	double meanv2;
	double sumlogdiag;
	double logw;
	double chol00;
	double chol10;
	double chol11;
	double chol20;
	double chol21;
	double chol22;
	double weight;
};