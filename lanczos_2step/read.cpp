#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <fstream>
#include <complex>
#include <unistd.h>
#include <omp.h>
// #include <mkl_types.h>
// #define MKL_Complex16 std::complex<double>
// #include <mkl.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen; 

typedef vector< vector < complex<double> > > double_vector_2d; 
typedef Matrix<complex<double>, Dynamic, Dynamic, RowMajor> Rmatrixxcd; 

// #include "stoch_reconfig.h"
#include "functions.h"
#include "metropolis.h"

int Nsite = 16; 
int Nhid = Nsite; 
int Nsample = 5000; 
double J2 = 0.0;
int print_to_screen = 1;
int print_to_file = 1;   
int ppow = 0; 
double pi = acos(-1.0);
int Nskip = 20;
int bin_size = 50; 
int id = 0; 

ofstream fp, fpw; 


int main(int argc, char* argv[])
{

	srand(time(NULL));

	string input;

	char ch; 
	while((ch = getopt(argc, argv, "p:f:s:t:i:j:k:b:d:")) != EOF)
	{
		switch(ch)
		{
			
			case 'p' : print_to_screen = atoi(optarg); 
			break; 
			case 'f' : print_to_file = atoi(optarg); 
			break; 
			case 'i' : input = string(optarg);
			break; 
			case 's' : Nsample = atoi(optarg); 
			break; 
			case 'j' : J2 = atof(optarg); 
			break; 
			case 'k' : Nskip = atoi(optarg); 
			break; 
			case 't' : ppow = atoi(optarg); 
			break;
			case 'b' : bin_size = atoi(optarg); 
			break;
			case 'd' : id = atoi(optarg); 
			break;


		}
	}


	if(input.size() == 0 )
	{
		cout<<"No input file"<<endl; 
		return 0; 
	}

	ifstream fpi; 
	fpi.open(input, ios::in);

	if(!fpi) 
	{
		cout<<"Input file doesn't exist"<<endl; 
		return 0; 
	}

	

	fpi>>Nsite>>Nhid; 

	// cout<<Nsite<<"  "<<Nhid<<endl; 

	MatrixXcd W(Nhid, Nsite) ; 
	double x, y; 

	for(int i=0;i<Nhid;i++)
		for(int j=0;j<Nsite;j++)
		{
			fpi>>x>>y;  
			W(i, j) = complex<double>(x, y) ;  
		}

	fpi.close();

	MatrixXi pp(Nsite, 8*Nsite);
	all_position(Nsite, pp);
	MatrixXcd ss = MatrixXcd::Zero(Nsite, Nsite);


	vector< complex<double> > ene(10, 0);
	double accept_rt; 

	double_vector_2d data; 

	measure_energy(W, Nhid, Nsite, Nsample, Nskip, J2, accept_rt, pp, ppow, data, bin_size);


	int len = 1; 
	if(ppow == 1) len = 3; 
	else if(ppow == 2) len = 5; 


	cout.precision(8); 


	ofstream fp; 
	fp.open(to_string(id) + "data_" + input );

	cout.precision(8); 
	cout<<fixed; 

	fp<<data.size()<<endl; 

	for(int i=0;i<data.size();i++)
	{
		for(int j=0;j<data[i].size(); j++)
			fp<<fixed<<data[i][j]<<" "; 

		fp<<endl; 
	}

	fp.close();




	/*
	complex<double> ssum = 0.0;

	int l = (int)sqrt(Nsite+0.01);

	// cout<<Nsite<<endl; 

	
	for(int i=0;i<Nsite;i++)
		for(int j=0;j<Nsite;j++)
		{
			double ix = i / l, iy = i % l; 
			double jx = j / l, jy = j % l; 

			ssum += real( ss(i,j) ) * exp( complex<double>(0, 1.0) * ((ix - jx) * pi + (iy - jy) * pi) ); 
		}	

	ssum /= (double)Nsite; 


	ofstream fp; 
	fp.open("energy_" + input );

	cout.precision(8); 

	fp<<"energy per site is: "<<ene[0]/(double)Nsite<<endl; 
	fp<<"standard error is: "<<ene[5]/(double)Nsite<<endl; 
	// fp<<real(ssum)<<" "<<imag(ssum)<<endl;
	fp<<"acceptance is: "<<accept_rt<<" Nskip is: "<<Nskip<<endl; 

	fp.close();


	
	fp.open("ssf_" + input );

	for(int i=0;i<Nsite;i++)
	{
		for(int j=0;j<Nsite;j++)
			fp<<real(ss(i, j))<<" "; 

		fp<<endl; 
	}

	fp.close();
	*/





	/*

	cout<<ene[0]/(double)Nsite<<endl; 

	double b = sqrt(real(ene[1]) - real(ene[0])*real(ene[0]) ); 

	cout<<"variance: "<<b<<endl; 

	
	

	double top = real(ene[2]) -  3.0 * real(ene[0])*real(ene[1]) + 2.0 * pow(real(ene[0]), 3.0); 
	double bot = 2.0 * pow( real(ene[1]) - real(ene[0])*real(ene[0]), 1.5); 
	double f = top / bot; 

	cout<<"f: "<<top<<" "<<bot<<" "<<f<<endl; 

	double a = f - sqrt(f*f + 1); 
	cout<<"alpha: "<<a<<endl;

	cout<<"new new_energy 1 step: "<<ene[0]<< "  "<< b * a<<"  "<<ene[0] + b*a<<endl; 


	double c1 = 1.0/sqrt(a*a + 1.0) - real(ene[0])*a/(b * sqrt(a*a + 1.0)); 
	double c2 = a/(b * sqrt(a*a + 1));


	cout<<"coefficient: "<<c1<<" "<<c2<<endl;
	
	complex<double> new_energy = 0.0; 

	new_energy = c1*c1 * ene[0] + 2.0*c1*c2*ene[1] + c2*c2*ene[2]; 

	cout<<"new_energy from measurement 1 step: "<<new_energy<<"  "<<new_energy/(double)Nsite<<endl; 


	double h = real(new_energy); 
	double hh = real(c1*c1*ene[1] + 2.0 * c1*c2*ene[2] + c2*c2*ene[3]); 
	double hhh = real(c1*c1*ene[2] + 2.0*c1*c2*ene[3] + c2*c2*ene[4]);  
	
	top = hhh -  3.0 * h*hh + 2.0 * pow(h, 3.0); 
	bot = 2.0 * pow(hh - h*h, 1.5); 
	f = top / bot; 

	cout<<"f: "<<top<<" "<<bot<<" "<<f<<endl; 

	b = sqrt(hh - h * h); 
	a = f - sqrt(f*f + 1); 
	cout<<"alpha: "<<a<<endl;

	cout<<"new new_energy 2 step: "<<h<< "  "<< b * a<<"  "<<h + b*a<<endl;
	cout<<(h + b*a)/(double)Nsite<<endl; 
	
	*/
	





	return 0; 

}