
#ifndef nat_grad_H
#define nat_grad_H


#include <iostream>
#include <fstream>
#include <string>

#include "mkl_types.h"
#define MKL_Complex16 std::complex<double>


//#include <chrono>
#include <omp.h>
#include <complex>
#include <cstdlib>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<vector>
#include "mkl.h"

using namespace std;



	void mat_vec_mult(complex<double>* & , complex<double>* & , complex<double>* & ,  complex<double>* & ,int , int , double );

	void Nat_Grad_CG(complex<double>* , complex<double>* , complex<double>* , complex<double>* , double & , int , int , int ,double ) ;
	void Nat_Grad_CG_mth(complex<double>* , complex<double>* , complex<double>* , complex<double>* , double & , int , int , int ,double ) ;







#endif
