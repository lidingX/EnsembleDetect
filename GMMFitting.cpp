#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <float.h>
#include <math.h>
// X: n*d data
// labels: n
void gmmfitting(int& components, double*& weights_, double*& logweights_, double* & sumlogdiags_, double*& means, double*& precisions_chol, double* X, int* labels, int n, int d, int init_k, double tol, int maxiter)
{
	int k = init_k;
	// allocating memory for tmp variables and output variables
	double* R = new double[k*n]; // k*n-matrix responsibility
	bool* saved = new bool[k];// k-array flags indicating whether corresponding mix should be omitted
	double* w = new double[k];// k-array weights of mix
	double* logw = new double[k];// k-array log(weights)
	double* cX = new double[n*d]; // n*d-matrix decentralized data
	double* m = new double[k*d]; // k*d-matrix means of mix
	double* cov = new double[k*d*d];// k*d*d-matrix  covariance of mix
	double* chol_prec = new double[k*d*d];// k*d*d-matrix  lower triangular matrix for cholesky decomostion of precision of mix
	double* tmpr = new double[k];//k-array tmp of responsibility
	double* sumlogdiag = new double[k];//k-array log(det(precision))
									   //initialization
	memset(R, 0, n*k * sizeof(double));
	for (int num = 0; num < n; num++)
	{
		R[labels[num] * n + num] = 1;
	}
	double llh = -DBL_MAX;
	double llh_;
	for (int i = 0; i < maxiter; i++)
	{
		memset(w, 0, k * sizeof(double));
		memset(m, 0, k*d * sizeof(double));
		memset(cov, 0, k*d*d * sizeof(double));
		memset(saved, false, k * sizeof(bool));
		for (int mix = 0; mix < k; mix++)
		{
			int R_mix_idx = mix*n;
			int m_mix_idx = mix*d;
			int cov_mix_idx = mix*d*d;
			//maximization:
			//weights
			for (int num = 0; num < n; num++)
			{
				w[mix] += R[R_mix_idx + num];
			}
			double div = 1 / w[mix];
			w[mix] /= n;
			logw[mix] = log(w[mix]);
			//printf("w:%d %f ", mix, w[mix]);
			// means
			for (int num = 0; num < n; num++)
			{
				int R_idx = R_mix_idx + num;
				double r = R[R_idx] * div;
				int ele_d_idx = num*d;
				for (int dim = 0; dim < d; dim++)
				{
					m[m_mix_idx + dim] += X[ele_d_idx + dim] * r;
				}
			}
			//printf("m:%d %f ", mix, m[m_mix_idx]);
			// decentralization
			for (int num = 0; num < n; num++)
			{
				int ele_d_idx = num*d;
				for (int dim = 0; dim < d; dim++)
				{
					cX[ele_d_idx + dim] = X[ele_d_idx + dim] - m[m_mix_idx + dim];
				}
			}
			// covariance
			for (int num = 0; num < n; num++)
			{
				int R_idx = R_mix_idx + num;
				double r = R[R_idx];
				int ele_d_idx = num*d;
				for (int dim1 = 0; dim1 < d; dim1++)
				{
					int cov_d_idx = cov_mix_idx + dim1*d;
					for (int dim2 = 0; dim2 < d; dim2++)
					{
						cov[cov_d_idx + dim2] += cX[ele_d_idx + dim1] * cX[ele_d_idx + dim2] * r*div;
					}
				}
			}
			//printf("c:%d %f ", mix, cov[cov_mix_idx ]);
			// regularization
			for (int dim1 = 0; dim1 < d; dim1++)
			{
				for (int dim2 = 0; dim2 < d; dim2++)
				{
					if (dim1 == dim2)
					{
						cov[cov_mix_idx + dim1*d + dim2] += 1e-6;
					}
				}
			}
			/*for (int dim = 0; dim < d*d; dim++)
			{
			printf("%2.10f ", cov[mix*d*d + dim]);
			}
			printf("\n");*/
			// cholesky decompostion of covariance, cov = L*L', L is a lower triangle
			for (int dim1 = 0; dim1 < d; dim1++)
			{
				for (int dim2 = 0; dim2 < d; dim2++)
				{
					if (dim1 == dim2)
					{
						double tmp = cov[cov_mix_idx + dim2*d + dim2];
						for (int kk = 0; kk < dim2; kk++)
						{
							double covdim2k = cov[cov_mix_idx + dim2*d + kk];
							tmp -= covdim2k * covdim2k;
						}
						cov[cov_mix_idx + dim2*d + dim2] = sqrt(tmp);
					}
					else if (dim1 > dim2)
					{
						double tmp = cov[cov_mix_idx + dim1*d + dim2];
						for (int kk = 0; kk < dim2; kk++)
						{
							tmp -= cov[cov_mix_idx + dim1*d + kk] * cov[cov_mix_idx + dim2*d + kk];
						}
						cov[cov_mix_idx + dim1*d + dim2] = tmp / cov[cov_mix_idx + dim2*d + dim2];
					}
				}
			}
			// cholesky decompostion of covariance, prec = inv(cov) = L_'*L_, L_ = inv(L)
			// computed by backsubsitution
			double sum = 0;
			for (int dim1 = 0; dim1 < d; dim1++)
			{
				for (int dim2 = 0; dim2 < d; dim2++)
				{
					if (dim1 == dim2)
					{
						double tmp = 1;
						for (int kk = dim2; kk < dim1; kk++)
						{
							tmp -= cov[cov_mix_idx + dim1*d + kk] * chol_prec[cov_mix_idx + kk*d + dim2];
						}
						double diag = tmp / cov[cov_mix_idx + dim1*d + dim1];
						chol_prec[cov_mix_idx + dim2*d + dim2] = diag;
						sum += log(diag);
					}
					else if (dim1 > dim2)
					{
						double tmp = 0;
						for (int kk = dim2; kk < dim1; kk++)
						{
							tmp -= cov[cov_mix_idx + dim1*d + kk] * chol_prec[cov_mix_idx + kk*d + dim2];
						}
						chol_prec[cov_mix_idx + dim1*d + dim2] = tmp / cov[cov_mix_idx + dim1*d + dim1];
					}
				}
			}
			// log(det(prec)) = sum(log(diag(chol(prec))))
			sumlogdiag[mix] = sum;
			// expectaion:
			// responsbility of mix
			for (int num = 0; num < n; num++)
			{
				//Mahalanobis distance
				double m_dis = 0;
				int ele_d_idx = num*d;
				for (int dim1 = 0; dim1 < d; dim1++)
				{
					int cov_d_idx = cov_mix_idx + dim1*d;
					double m_dis_dim1 = 0;
					for (int dim2 = 0; dim2 < d; dim2++)
					{
						if (dim1 >= dim2)
						{
							m_dis_dim1 += chol_prec[cov_d_idx + dim2] * cX[ele_d_idx + dim2];
						}
					}
					m_dis += m_dis_dim1*m_dis_dim1;
				}
				//log likelihoodlog likelihood, constant omitted
				double r = -m_dis / 2 + sumlogdiag[mix] + logw[mix];
				R[R_mix_idx + num] = r;
			}
		}
		// logsumexp
		llh_ = llh;
		llh = 0;
		for (int num = 0; num < n; num++)
		{
			double maxr = -DBL_MAX;
			int id = 0;
			for (int mix = 0; mix < k; mix++)
			{
				double r = R[mix*n + num];
				tmpr[mix] = r;
				if (r > maxr)
				{
					maxr = r;
					id = mix;
				}
			}
			saved[id] = true;
			double sum = 0;
			for (int mix = 0; mix < k; mix++)
			{
				sum += exp(tmpr[mix] - maxr);
			}
			double logsum = log(sum);
			for (int mix = 0; mix < k; mix++)
			{
				//printf("%f\n", exp(*r - logsum - maxr));
				R[mix*n + num] = exp(tmpr[mix] - logsum - maxr);
			}
			llh += logsum + maxr;
		}
		llh /= n;
		/*printf("R:\n");
		for (int num = 0; num < n; num++)
		{
		for (int mix = 0; mix < k; mix++)
		{
		printf("%lf ", R[mix*n + num]);
		}
		printf("\n");
		}*/
		// convergence checking
		/*if (fabs(llh - llh_) < tol*fabs(llh))
		{
		printf("iter:%d llh:%f llh_:%f\n", i, llh, llh_);
		break;
		}*/
		printf("iter:%d llh:%f\n", i, llh);
		// empty cluster elimination
		int k_ = 0;
		for (int mix = 0; mix < k; mix++)
		{
			if (saved[mix])
			{
				k_++;
			}
		}
		if (k_ != k)
		{
			//printf("iter:%d %d llh:%f llh_:%f\n",k_, i, llh, llh_);
			printf("elimination iter:%d k_:%d %f\n", i, k_, llh);
			double * R_ = new double[n*k_];
			double sumr = 0;
			int mix_ = 0;
			for (int mix = 0; mix < k; mix++)
			{
				if (saved[mix])
				{
					for (int num = 0; num < n; num++)
					{
						double r = R[mix*n + num];
						R_[mix_*n + num] = r;
						sumr += r;
					}
					mix_++;
				}
			}
			k = k_;
			delete[] R;
			R = R_;
			for (int mix = 0; mix < k; mix++)
			{
				for (int num = 0; num < n; num++)
				{
					R_[mix*n + num] /= sumr;
				}
			}
		}
	}
	delete[] R;
	delete[] tmpr;
	delete[] saved;
	delete[] cov;
	delete[] cX;
	components = k;
	weights_ = w;
	logweights_ = logw;
	means = m;
	precisions_chol = chol_prec;
	sumlogdiags_ = sumlogdiag;
}