#pragma once
void gmmfitting(int& components, double*& weights_, double*& logweights_, double* & sumlogdiags_, double*& means, double*& precisions_chol, double* X, int* labels, int n, int d, int init_k, double tol, int maxiter);
