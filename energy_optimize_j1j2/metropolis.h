#ifndef vmc_part
#define vmc_part

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen; 

typedef vector< vector < complex<double> > > double_vector_2d; 


complex<double> metropolis(MatrixXcd& W, double lamda, int Nhid,  int Nsite, int Nsample, double J2, int Nskip, double& accept_rt, double& rtol, MatrixXi& pp); 

// void measure_energy(MatrixXcd& W, int Nhid, int Nsite, int avg_sample, double J1, double J2, complex<double>& ene, double& error);


#endif