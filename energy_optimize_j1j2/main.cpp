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
#include <mkl_types.h>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
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
int train_sample = 5000; 
int avg_sample = 100000; 
double Wscale = 0.5;
int MCsteps = 101; 
double J2 = 0.0;
double lamda = 0.01;  
double mu = 0.1;
int print_to_screen = 1;
int print_to_file = 1;  
int Nskip = 20;
int trial = 10; 



ofstream fp, fpw, fpe; 

bool cmp(const pair<double, MatrixXcd>& x_, const pair<double, MatrixXcd>& y_)
{
	return x_.first < y_.first; 
}


MatrixXcd Ground_state(MatrixXcd& W, string name)
{

	MatrixXcd vW(Nhid, Nsite), dW(Nhid, Nsite);
	complex<double> ground_state_energy = 0; 
	double accept_rt, rtol;

	MatrixXi pp(Nsite, 8*Nsite);

	all_position(Nsite, pp);

	// cout<<pp.row(0)<<endl; 


	vector<pair<double, MatrixXcd> > gw; 

	for (int t=0;t<MCsteps;t++)
	{

		complex<double> current_energy = metropolis(W, lamda, Nhid, Nsite, train_sample, J2, Nskip, accept_rt, rtol, pp); 

		Nskip = min(100, (int)((1.0/accept_rt)*5));
		// Nskip = 20;


		if(real(current_energy)/(double)Nsite < -0.46)
		{
			gw.push_back(make_pair(real(current_energy), W)); 
		}

		if(t%5==0)
		{
			if(print_to_screen) cout<<t<<' '<<current_energy/(double)Nsite <<", acceptance is "<<accept_rt<<", tolerance is "<<rtol<<", next Nskip is "<<Nskip<<endl; 
			if(print_to_file) fp<<t<<' '<<current_energy/(double)Nsite <<", acceptance is "<<accept_rt<<", tolerance is "<<rtol<<", next Nskip is "<<Nskip<<endl;
		}

		//save W matrix very 10 steps
		if(t%10 == 0 and print_to_file and t)
		{

			fpw.open(name); 

			fpw<<Nsite<<' '<<Nhid<<endl; 

			for(int i=0;i<Nhid;i++)
			{
				for(int j=0;j<Nsite;j++)
					fpw<<real(W(i,j))<<' '<<imag(W(i,j))<< (j == Nsite - 1 ? '\n':' '); 
			}

			fpw.close(); 

		}


	}

	sort(gw.begin(), gw.end(), cmp); 

	//save matrix to be evaluated

	for(int t=0;t < trial and t < gw.size();t++)
	{

		fpe<<Nsite<<' '<<Nhid<<endl; 
		for(int i=0;i<Nhid;i++)
		{
			for(int j=0;j<Nsite;j++)
				fpe<<real(gw[t].second(i,j))<<' '<<imag(gw[t].second(i,j))<< (j == Nsite - 1 ? '\n':' '); 
		}
	}


	/*

	complex<double> ene; 
	double error; 

	for(int i=0;i< trial and i < gw.size();i++)
	{
		// cout<<gw[i].first<<endl; 
		measure_energy(gw[i].second, Nhid, Nsite, avg_sample,  J1, J2, ene, error);

		if(real(ene) < real(ground_state_energy))
		{
			ground_state_energy = ene; 
			ground_state_W = gw[i].second; 
		}

		
		if(print_to_screen) 
			cout<<i<<" "<<avg_sample<<" average energy is: "<< ene/(double)Nsite<<", error is: "<<error/(double)Nsite<<" original energy is "<<gw[i].first/(double)Nsite<<endl; 
		if(print_to_file)
			fp<<i<<" "<<avg_sample<<" average energy is: "<< ene/(double)Nsite<<", error is: "<<error/(double)Nsite<<" original energy is "<<gw[i].first/(double)Nsite<<endl;
	}


	if(print_to_screen) 
		cout<<"ground state energy is "<< ground_state_energy/(double)Nsite<<endl; 
	if(print_to_file)
		fp<<"ground state energy is "<< ground_state_energy/(double)Nsite<<endl;
	*/

	return W; 

}




int main(int argc, char* argv[])
{

	srand(time(NULL));

	char ch; 
	while((ch = getopt(argc, argv, "n:l:m:p:f:s:t:w:j:a:h:")) != EOF)
	{
		switch(ch)
		{
			case 'n' :Nsite = atoi(optarg), Nhid = Nsite;
			break; 
			case 'l' : lamda = atof(optarg); 
			break; 
			case 'm' : MCsteps = atoi(optarg); 
			break; 
			case 'w' : Wscale = atof(optarg);
			break; 
			case 'p' : print_to_screen = atoi(optarg); 
			break; 
			case 'f' : print_to_file = atoi(optarg); 
			break; 
			case 's' : train_sample = atoi(optarg); 
			break; 
			case 'a' : avg_sample = atoi(optarg); 
			break; 
			case 't' : trial = atoi(optarg); 
			break; 
			case 'j' : J2 = atof(optarg); 
			break; 
			case 'h' : Nhid = int( atof(optarg) * Nsite ); 
			break;

		}
	}


	string name, tail, wname; 

	string s_w = to_string(Wscale); 
	while(s_w[ s_w.size() - 1] == '0') s_w.pop_back(); 

	string s_j = ""; 
	if(J2 != 0.0)
	{
		s_j = to_string(J2); 
		while(s_j[ s_j.size() - 1] == '0') s_j.pop_back();
		s_j += "_";
	}

	string s_lamda = to_string(lamda); 
	while(s_lamda[ s_lamda.size() - 1] == '0') s_lamda.pop_back(); 

	tail = to_string(Nsite) + "_" + to_string(Nhid) + "_" + s_j + to_string(train_sample)+"_" + to_string(MCsteps) + "_" + s_w + "_" + s_lamda + ".txt"; 
	

	
	bool flag = false; 
	//check whether there is already a w matrix

	ifstream check; 
	check.open("wmatrix_" + tail); 
	if(check) 
	{
		flag = true; 
		
		if(print_to_screen) cout<<"file exist"<<endl; 
	}
	check.close(); 


	MatrixXcd W(Nhid, Nsite); 	

	wname = "wmatrix_"  + tail; //with the original lamda

	if(check)
	{

		char* tname = new char [wname.length()+1];
		strcpy(tname, wname.c_str());
		freopen(tname, "r", stdin);

		scanf("%d%d", &Nsite, &Nhid); 

		double x, y; 

		for(int i=0;i<Nhid;i++)
			for(int j=0;j<Nsite;j++)
			{
				scanf("%lf%lf", &x, &y); 
				W(i, j) = complex<double>(x, y) ;
			}

		// cout<<W<<endl; 

		fclose(stdin); 


	}
	else 
	{
		for (int i=0;i<Nhid;i++)
		{
			for(int j=0;j<Nsite;j++)
				W(i, j) = complex<double>(Wscale * (random_number() - 0.5), Wscale * (random_number() - 0.5)); 
		}
	}


	// lamda *= 0.2;
	s_lamda = to_string(lamda); 
	while(s_lamda[ s_lamda.size() - 1] == '0') s_lamda.pop_back(); 

	tail = to_string(Nsite) + "_" + to_string(Nhid) + "_" + s_j + to_string(train_sample)+"_" + to_string(MCsteps) + "_" + s_w + "_" + s_lamda + ".txt";


	if(print_to_file)
	{
		name = "monitor_" + tail; 
		fp.open(name, ios::app);

		name = "evaluate_" + tail; 
		fpe.open(name, ios::app); 
	}
	


	MatrixXcd ground_state_W(Nhid, Nsite); 

	double time = omp_get_wtime(); 
	
	ground_state_W = Ground_state(W, wname); 


	cout<<omp_get_wtime() - time<<"s"<<endl; 

	/*

	Nsite = 16;

	MatrixXi pp(Nsite, 8*Nsite);
	MatrixXd states(8*Nsite, Nsite);

	int x = 12;

	

	

	// cout<<pp.row(0)<<endl; 

	

	// cout<<states<<endl; 
	for(int x =0;x<Nsite;x++)
	{
		all_position(x, pp);

		VectorXd st = VectorXd::Zero(Nsite);
		st[x] = 1.0;

		translational_symmetry(st, states);

		for(int i=0;i<8*Nsite;i++)
		if(states(i, pp(0, i)) != 1.0 )
		{
			cout<<"error"<<endl; 
			cout<<states.row(i)<<endl; 
		}
	}
	*/
	


	

	fp.close(); 
	fpe.close(); 


	





	return 0; 

}