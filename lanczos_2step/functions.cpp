#include "functions.h"
#include <iostream>
double PI = acos(-1.0); 
// int Nsite = 8;


double random_number()
{
	double r = (double)(rand()/ (double)RAND_MAX);
    return r;
}


void all_position(int Nsite, MatrixXi& pp)
{

	int N = (int)sqrt(Nsite+0.01);

	MatrixXi pos(8, 2);

	int xx, yy;

	for(int i=0;i<Nsite;i++)
	{
		int x = i/N, y = i%N;

		pos(0, 0) = x , pos(0, 1) = N-1-y; 
		pos(1, 0) = N-1-x , pos(1, 1) = y; 
		pos(2, 0) = y , pos(2, 1) = x;
		pos(3, 0) = N-1-y , pos(3, 1) = N-1-x;

		for(int j=4;j<8;j++)
		{
			xx = N-1-y, yy=x; 
			pos(j, 0) = xx, pos(j, 1) = yy; 

			x = xx, y = yy; 
		}

		// for(int j=0;j<8;j++)
		// 	cout<<pos(j, 0)<<" "<<pos(j, 1)<<endl;


		for(int j=0;j<8;j++)
		{
			// int x1 = i/N, y1 = i%N;
			int x1 = pos(j, 0), y1 = pos(j, 1);

			for(int k=0;k<N;k++)
			{
				int ty1 = y1; 

				for(int s=0;s<N;s++)
				{
					pp(i, j*Nsite + k*N+s) = x1 * N + (ty1 - s + N) % N; 
				}
				
				x1 = (x1 + 1) %N; 
			}
		}

		

	} 

}

void translational_position(int p1, int p2, int Nsite, MatrixXi& tpos)
{
	int N = (int)sqrt(Nsite); 

	int x1 = p1 / N, y1 = p1 % N; 
	int x2 = p2 / N, y2 = p2 % N; 


	for(int k=0;k<N;k++)
	{
		int ty1 = y1, ty2 = y2; 

		for(int s=0;s<N;s++)
		{
			tpos(k*N+s, 0) = x1 * N + (ty1 - s + N) % N; 
			tpos(k*N+s, 1) = x2 * N + (ty2 - s + N) % N;
		}
		
		x1 = (x1 + 1) %N; 
		x2 = (x2 + 1) %N;

	}
}





void translational_symmetry(VectorXd& state, MatrixXd& all_states)
{

	int N = (int)sqrt( state.size() ), len = state.size(); 

	VectorXd tmp(N);

	MatrixXd mat(N, N), S(N, N), SS(N, N); 

	vector<MatrixXd> vv;

	for(int i=0;i<state.size();i++)
		mat(i/N, i%N) = state[i]; 
	

	S = mat;

	//reflection symmetry

	SS = S.rowwise().reverse();
	vv.push_back(SS); // axis : y 0

	SS = S.colwise().reverse();
	vv.push_back(SS); //axis: x 1

	SS = S.transpose();
	vv.push_back(SS); // axis y=-x 2

	for(int i=0;i<N;i++)
	{
		tmp = S.row(i);
		SS.col(N-1-i) = tmp;
	}
	S = SS.colwise().reverse();

	vv.push_back(S); // axis y=x, 3

	//rotation symmetry
	S = mat;  //important

	for(int i=0;i<4;i++)
	{
		SS = S.transpose().colwise().reverse(); 
		vv.push_back(SS);
		S = SS;
	}

	// for(int t=0;t<vv.size();t++)
	// 	cout<<vv[t]<<endl<<endl; 


	

	for(int t=0;t<vv.size();t++)
	{
		mat = vv[t];

		for(int k=0;k<N;k++)
		{
			for(int s=0;s<N;s++)
			{

				VectorXd all(N*N); 
				int cnt = 0; 
				for(int i=0;i<N;i++)
				{
					for(int j=s;j<N;j++)
						all(cnt++) = mat(i, j); 
					for(int j=0;j<s;j++)
						all(cnt++) = mat(i, j); 
				}

				all_states.row(t * len + k*N+s) = all; 

			}
			
			tmp = mat.row(N-1);
			for(int i=N-1;i>0;i--)
				mat.row(i) = mat.row(i-1); 
			mat.row(0) = tmp; 

			// cout<<mat<<endl<<endl;
		}


	}




	

	// cout<<all_states<<endl; 

}


void coeff_partial(MatrixXd& states, MatrixXcd& W, Rmatrixxcd& partial, int Nhid, MatrixXcd& temp, VectorXcd& coeff)
{

	int N = coeff.size(); //8*Nsite
	int Nsite = states.cols();

	VectorXcd tmp(Nhid), tanh_prod(Nhid);
	VectorXd st(Nsite);  

	complex<double> ssum = 0.0; 

	Rmatrixxcd tmp_partial = Rmatrixxcd::Zero(Nhid, Nsite); 

	for(int i=0;i<N;i++)
	{
		st = states.row(i); 
		tmp = temp.row(i);

		ssum += coeff[i];

		tanh_prod = tanh(tmp.array());
		tmp_partial = tanh_prod * st.transpose(); 
		tmp_partial *= coeff[i]; 
		// cout<<tmp_partial.size()<<endl;
		partial += tmp_partial;
		// cout<<"test_part"<<endl; 
	}	
	
	partial /= ssum; 
}

complex<double> single_coeff(VectorXd& state, MatrixXcd& W, VectorXcd& tp)
{
	tp = W * state; 
	return cosh(tp.array()).prod(); 
}


complex<double> coefficient(MatrixXd& states, MatrixXcd& W, MatrixXcd& temp, VectorXcd& coeff)
{
	int N = coeff.size(); //8*Nsite
	int Nsite = states.cols();

	// cout<<N<<endl; 

	VectorXd st(Nsite); 

	complex<double> res(0, 0);
	int id; 

	for(int i=0;i<N;i++)
	{
		id = i/Nsite; 
		st = states.row(i);
		temp.row(i) = W * st;
		coeff[i] = cosh(temp.row(i).array()).prod();

		// if(id == 2 or id == 3 or id == 4 or id == 6)
		// 	coeff[i] *= -1.0;

		res += coeff[i]; 
	}
	
	res /= (double)N; 
	// temp = cosh((W * state).array());

	return res;
}


complex<double> Get_energy_a1(VectorXd& state, MatrixXd& states, complex<double> sum_coeff, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp)
{

	complex<double> sz(0, 0); 
	complex<double> sxsy(0, 0);

	complex<double> res(0, 0);
	complex<double> coeff_new;
	int len = (int)sqrt(Nsite+0.01); 
	int id;
	// sum_coeff *= (double)(8*Nsite);

	// MatrixXcd temp(8*Nsite, Nhid);
	// VectorXcd coeff(8*Nsite);

	// MatrixXd states(8*Nsite, Nsite);




	for(int i=0;i<len;i++)
		for(int j=0;j<len;j++)
		{


			sz = 0.0, sxsy = 0.0; 

			int pos = i * len + j; 
			int right = i * len + ((j+1)%len); 
			int bottom = ( (i+1)%len ) * len + j;

			sz += state[pos] * state[right];
			sz += state[pos] * state[bottom];


			if(state[pos] * state[right] < 0.0)
			{

				coeff_new = complex<double>(0.0, 0.0); //important

				for(int ii=0;ii<8*Nsite;ii++)
				{
					complex<double> prod(1.0, 0); 
					id = ii/Nsite; 

					for(int jj=0;jj<Nhid;jj++)
					{
						prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) );
					}

					// if(id == 2 or id == 3 or id == 4 or id == 6)
					// 	prod *= -1.0;

					coeff_new += prod; 
				} 

				coeff_new /= (double)(8*Nsite);

				sxsy += coeff_new/sum_coeff;
				 
			}

			if(state[pos] * state[bottom] < 0.0)
			{
				coeff_new = complex<double>(0.0, 0.0); //important

				for(int ii=0;ii<8*Nsite;ii++)
				{
					complex<double> prod(1.0, 0); 
					id = ii/Nsite; 

					for(int jj=0;jj<Nhid;jj++)
					{
						prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) );
					}

					// if(id == 2 or id == 3 or id == 4 or id == 6)
					// 	prod *= -1.0;

					coeff_new += prod; 
				} 

				coeff_new /= (double)(8*Nsite);
				
				sxsy += coeff_new/sum_coeff;
			}


			res += (sz + 0.5*sxsy);


			if(J2 != 0.0)
			{
				sz = 0.0, sxsy = 0.0; 

				right = ((i+1)%len) * len + ((j+1)%len); 
				bottom = ( (i+1)%len ) * len + ( (j-1+len)%len ); 

				sz += state[pos] * state[right] ;
				sz += state[pos] * state[bottom] ; 

				if(state[pos] * state[right] < 0.0)
				{

					coeff_new = complex<double>(0.0, 0.0); //important

					for(int ii=0;ii<8*Nsite;ii++)
					{
						complex<double> prod(1.0, 0); 
						id = ii/Nsite; 

						for(int jj=0;jj<Nhid;jj++)
						{
							prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) );
						}

						// if(id == 2 or id == 3 or id == 4 or id == 6)
						// 	prod *= -1.0;

						coeff_new += prod; 
					} 

					coeff_new /= (double)(8*Nsite);

					sxsy += coeff_new/sum_coeff;
				}

				if(state[pos] * state[bottom] < 0.0)
				{

					coeff_new = complex<double>(0.0, 0.0); //important

					for(int ii=0;ii<8*Nsite;ii++)
					{
						complex<double> prod(1.0, 0); 
						id = ii/Nsite; 

						for(int jj=0;jj<Nhid;jj++)
						{
							prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) );
						}

						// if(id == 2 or id == 3 or id == 4 or id == 6)
						// 	prod *= -1.0;

						coeff_new += prod; 
					} 

					coeff_new /= (double)(8*Nsite);
					
					sxsy += coeff_new/sum_coeff;

				}

				res += J2*(sz + 0.5 * sxsy); 

			}
		

		}


		return res;

}



