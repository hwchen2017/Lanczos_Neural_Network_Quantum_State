#ifndef vmc_part
#define vmc_part

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen; 

typedef vector< vector < complex<double> > > double_vector_2d;


void measure_energy(MatrixXcd& W, int Nhid, int Nsite, int Nsample, int Nskip, double J2, double& accept_rt, 
	MatrixXi& pp, int ppow, double_vector_2d& data, int bin_size);

complex<double> Hntr_a1(VectorXd& state, MatrixXd& states, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp); 

complex<double> HHntr_a1(VectorXd& state, MatrixXd& states, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp);

complex<double> HHHntr_a1(VectorXd& state, MatrixXd& states, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp);

#endif