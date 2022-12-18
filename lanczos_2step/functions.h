#ifndef functions
#define functions 

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
typedef Matrix<complex<double>, Dynamic, Dynamic, RowMajor> Rmatrixxcd; 




	
// void stoch_reconfig(const MatrixXcd& A, const VectorXcd& der_avg, const VectorXcd& grad, VectorXcd& x,  
// 	int Npar, int Nsample, int Nit, double tol, double lam); 

void coeff_partial(MatrixXd& states, MatrixXcd& W, Rmatrixxcd& partial, int Nhid, MatrixXcd& temp, VectorXcd& coeff);

complex<double> coefficient(MatrixXd& states, MatrixXcd& W, MatrixXcd& temp, VectorXcd& coeff); 

void all_position(int Nsite, MatrixXi& pp);

void translational_position(int p1, int p2, int Nsite, MatrixXi& tpos);

void translational_symmetry(VectorXd& state, MatrixXd& all_states);

complex<double> single_coeff(VectorXd& state, MatrixXcd& W, VectorXcd& tp);

complex<double> Get_energy(MatrixXd& states, complex<double> sum_coeff, MatrixXcd& W,  int Nhid, int Nsite, double J2, MatrixXcd& temp, VectorXcd& coeff); 

complex<double> Get_energy_a1(VectorXd& state, MatrixXd& states, complex<double> sum_coeff, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp);


double random_number();



#endif