#include <iostream>
#include <ctime>
#include "metropolis.h"
// #include "stoch_reconfig.h"
#include "functions.h"


complex<double> Hntr_a1(VectorXd& state, MatrixXd& states, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp)
{

	complex<double> sz(0, 0); 
	complex<double> sxsy(0, 0);

	complex<double> res(0, 0);
	complex<double> coeff_new;
	int len = (int)sqrt(Nsite+0.01); 
	int id;

	complex<double> self_coeff = complex<double>(0.0, 0.0);


	for(int i=0;i<8*Nsite;i++)
	{

		complex<double> prod(1.0, 0); 

		for(int j=0;j<Nhid;j++)
			prod *= cosh(temp(i, j));
 
		self_coeff += prod; 
	} 

	self_coeff /= (double)(8*Nsite);



	for(int i=0;i<len;i++)
		for(int j=0;j<len;j++)
		{


			sz = 0.0, sxsy = 0.0; 

			int pos = i * len + j; 
			int right = i * len + ((j+1)%len); 
			int bottom = ( (i+1)%len ) * len + j;


			sz += (state[pos] * state[right] + state[pos] * state[bottom]) * self_coeff; 

			// sz += state[pos] * state[right];
			// sz += state[pos] * state[bottom];


			if(state[pos] * state[right] < 0.0)
			{
				//right
				coeff_new = complex<double>(0.0, 0.0); //important

				for(int ii=0;ii<8*Nsite;ii++)
				{
					complex<double> prod(1.0, 0);  

					for(int jj=0;jj<Nhid;jj++)
					{
						prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) );
					}

					coeff_new += prod; 
				} 

				coeff_new /= (double)(8*Nsite);

				sxsy += coeff_new;
				 
			}

			if(state[pos] * state[bottom] < 0.0)
			{

				//bottom
				coeff_new = complex<double>(0.0, 0.0); //important

				for(int ii=0;ii<8*Nsite;ii++)
				{
					complex<double> prod(1.0, 0); 

					for(int jj=0;jj<Nhid;jj++)
					{
						prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) );
					}

					coeff_new += prod; 
				} 

				coeff_new /= (double)(8*Nsite);
				
				sxsy += coeff_new;
				 
			}


			res += (sz + 0.5*sxsy);


			if(J2 != 0.0)
			{
				sz = 0.0, sxsy = 0.0; 

				right = ((i+1)%len) * len + ((j+1)%len); 
				bottom = ( (i+1)%len ) * len + ( (j-1+len)%len ); 

				sz += (state[pos] * state[right] + state[pos] * state[bottom]) * self_coeff;

				// sz += state[pos] * state[right] ;
				// sz += state[pos] * state[bottom] ; 

				if(state[pos] * state[right] < 0.0)
				{

					//right
					coeff_new = complex<double>(0.0, 0.0); //important

					for(int ii=0;ii<8*Nsite;ii++)
					{
						complex<double> prod(1.0, 0);  

						for(int jj=0;jj<Nhid;jj++)
						{
							prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) );
						}

						coeff_new += prod; 
					} 

					coeff_new /= (double)(8*Nsite);

					sxsy += coeff_new;
				}

				if(state[pos] * state[bottom] < 0.0)
				{

					//bottom
					coeff_new = complex<double>(0.0, 0.0); //important

					for(int ii=0;ii<8*Nsite;ii++)
					{
						complex<double> prod(1.0, 0); 

						for(int jj=0;jj<Nhid;jj++)
						{
							prod *= cosh(temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) );
						}

						coeff_new += prod; 
					} 

					coeff_new /= (double)(8*Nsite);
					
					sxsy += coeff_new;

				}

				res += J2*(sz + 0.5 * sxsy); 

			}
		

		}


		return res;

}




complex<double> HHntr_a1(VectorXd& state, MatrixXd& states, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp)
{

	complex<double> sz(0, 0); 
	complex<double> sxsy(0, 0);

	complex<double> res(0, 0);
	complex<double> coeff_new;
	int len = (int)sqrt(Nsite+0.01); 
	int id;

	MatrixXcd mv_prod_flip(8*Nsite, Nhid);
	MatrixXd new_state_tran(8*Nsite, Nsite);

	new_state_tran = states; 
	mv_prod_flip = temp; 

	complex<double> self_Hntr = Hntr_a1(state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);


	for(int i=0;i<len;i++)
		for(int j=0;j<len;j++)
		{


			sz = 0.0, sxsy = 0.0; 

			int pos = i * len + j; 
			int right = i * len + ((j+1)%len); 
			int bottom = ( (i+1)%len ) * len + j;

			sz += (state[pos] * state[right] + state[pos] * state[bottom]) *  self_Hntr; 

			// sz += state[pos] * state[right];
			// sz += state[pos] * state[bottom];


			if(state[pos] * state[right] < 0.0)
			{
				//right
				VectorXd new_state(Nsite); 
				new_state = state; 
				new_state[pos] *= -1.0; 
				new_state[right] *= -1.0;
				
				new_state_tran = states;

				for(int ii=0;ii<8*Nsite;ii++)
				{	

					new_state_tran(ii, pp(pos, ii)) *= -1.0;
					new_state_tran(ii, pp(right, ii)) *= -1.0;

					for(int jj=0;jj<Nhid;jj++)
					{
						mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) ;
					}
				} 


				sxsy += Hntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);
				 
			}

			if(state[pos] * state[bottom] < 0.0)
			{

				//bottom
				VectorXd new_state(Nsite); 
				new_state = state; 
				new_state[pos] *= -1.0; 
				new_state[bottom] *= -1.0;
				
				new_state_tran = states;

				for(int ii=0;ii<8*Nsite;ii++)
				{	

					new_state_tran(ii, pp(pos, ii)) *= -1.0;
					new_state_tran(ii, pp(bottom, ii)) *= -1.0;

					for(int jj=0;jj<Nhid;jj++)
					{
						mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) ;
					}
				} 


				sxsy += Hntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);
				 
			}


			res += (sz + 0.5*sxsy);


			if(J2 != 0.0)
			{
				sz = 0.0, sxsy = 0.0; 

				right = ((i+1)%len) * len + ((j+1)%len); 
				bottom = ( (i+1)%len ) * len + ( (j-1+len)%len ); 

				sz += (state[pos] * state[right] + state[pos] * state[bottom]) * self_Hntr;

				// sz += state[pos] * state[right] ;
				// sz += state[pos] * state[bottom] ; 

				if(state[pos] * state[right] < 0.0)
				{

					//right
					VectorXd new_state(Nsite); 
					new_state = state; 
					new_state[pos] *= -1.0; 
					new_state[right] *= -1.0;
					
					new_state_tran = states;

					for(int ii=0;ii<8*Nsite;ii++)
					{	

						new_state_tran(ii, pp(pos, ii)) *= -1.0;
						new_state_tran(ii, pp(right, ii)) *= -1.0;

						for(int jj=0;jj<Nhid;jj++)
						{
							mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) ;
						}
					} 


					sxsy += Hntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);
				}

				if(state[pos] * state[bottom] < 0.0)
				{

					VectorXd new_state(Nsite); 
					new_state = state; 
					new_state[pos] *= -1.0; 
					new_state[bottom] *= -1.0;
					
					new_state_tran = states;

					for(int ii=0;ii<8*Nsite;ii++)
					{	

						new_state_tran(ii, pp(pos, ii)) *= -1.0;
						new_state_tran(ii, pp(bottom, ii)) *= -1.0;

						for(int jj=0;jj<Nhid;jj++)
						{
							mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) ;
						}
					} 


					sxsy += Hntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);

				}

				res += J2*(sz + 0.5 * sxsy); 

			}
		

		}


		return res;

}

complex<double> HHHntr_a1(VectorXd& state, MatrixXd& states, MatrixXcd& W, int Nhid, int Nsite, double J2, MatrixXcd& temp, MatrixXi& pp)
{

	complex<double> sz(0, 0); 
	complex<double> sxsy(0, 0);

	complex<double> res(0, 0);
	complex<double> coeff_new;
	int len = (int)sqrt(Nsite+0.01); 
	int id;

	MatrixXcd mv_prod_flip(8*Nsite, Nhid);
	MatrixXd new_state_tran(8*Nsite, Nsite);

	new_state_tran = states; 
	mv_prod_flip = temp; 

	complex<double> self_HHntr = HHntr_a1(state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);


	for(int i=0;i<len;i++)
		for(int j=0;j<len;j++)
		{


			sz = 0.0, sxsy = 0.0; 

			int pos = i * len + j; 
			int right = i * len + ((j+1)%len); 
			int bottom = ( (i+1)%len ) * len + j;

			sz += (state[pos] * state[right] + state[pos] * state[bottom]) *  self_HHntr; 

			// sz += state[pos] * state[right];
			// sz += state[pos] * state[bottom];


			if(state[pos] * state[right] < 0.0)
			{
				//right
				VectorXd new_state(Nsite); 
				new_state = state; 
				new_state[pos] *= -1.0; 
				new_state[right] *= -1.0;
				
				new_state_tran = states;

				for(int ii=0;ii<8*Nsite;ii++)
				{	

					new_state_tran(ii, pp(pos, ii)) *= -1.0;
					new_state_tran(ii, pp(right, ii)) *= -1.0;

					for(int jj=0;jj<Nhid;jj++)
					{
						mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) ;
					}
				} 


				sxsy += HHntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);
				 
			}

			if(state[pos] * state[bottom] < 0.0)
			{

				//bottom
				VectorXd new_state(Nsite); 
				new_state = state; 
				new_state[pos] *= -1.0; 
				new_state[bottom] *= -1.0;
				
				new_state_tran = states;

				for(int ii=0;ii<8*Nsite;ii++)
				{	

					new_state_tran(ii, pp(pos, ii)) *= -1.0;
					new_state_tran(ii, pp(bottom, ii)) *= -1.0;

					for(int jj=0;jj<Nhid;jj++)
					{
						mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) ;
					}
				} 


				sxsy += HHntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);
				 
			}


			res += (sz + 0.5*sxsy);


			if(J2 != 0.0)
			{
				sz = 0.0, sxsy = 0.0; 

				right = ((i+1)%len) * len + ((j+1)%len); 
				bottom = ( (i+1)%len ) * len + ( (j-1+len)%len ); 

				sz += (state[pos] * state[right] + state[pos] * state[bottom]) * self_HHntr;

				// sz += state[pos] * state[right] ;
				// sz += state[pos] * state[bottom] ; 

				if(state[pos] * state[right] < 0.0)
				{

					//right
					VectorXd new_state(Nsite); 
					new_state = state; 
					new_state[pos] *= -1.0; 
					new_state[right] *= -1.0;
					
					new_state_tran = states;

					for(int ii=0;ii<8*Nsite;ii++)
					{	

						new_state_tran(ii, pp(pos, ii)) *= -1.0;
						new_state_tran(ii, pp(right, ii)) *= -1.0;

						for(int jj=0;jj<Nhid;jj++)
						{
							mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(right, ii))*states(ii, pp(right, ii)) ) ;
						}
					} 


					sxsy += HHntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);
				}

				if(state[pos] * state[bottom] < 0.0)
				{

					VectorXd new_state(Nsite); 
					new_state = state; 
					new_state[pos] *= -1.0; 
					new_state[bottom] *= -1.0;
					
					new_state_tran = states;

					for(int ii=0;ii<8*Nsite;ii++)
					{	

						new_state_tran(ii, pp(pos, ii)) *= -1.0;
						new_state_tran(ii, pp(bottom, ii)) *= -1.0;

						for(int jj=0;jj<Nhid;jj++)
						{
							mv_prod_flip(ii, jj) = temp(ii, jj) - 2.0*( W(jj, pp(pos, ii))*states(ii, pp(pos, ii)) +  W(jj, pp(bottom, ii))*states(ii, pp(bottom, ii)) ) ;
						}
					} 


					sxsy += HHntr_a1(new_state, new_state_tran, W, Nhid, Nsite, J2, mv_prod_flip, pp);

				}

				res += J2*(sz + 0.5 * sxsy); 

			}
		

		}


		return res;

}


