#include <iostream>
#include <ctime>
#include "metropolis.h"
// #include "stoch_reconfig.h"
#include "functions.h"




void measure_energy(MatrixXcd& W, int Nhid, int Nsite, int Nsample, int Nskip, double J2, double& accept_rt, MatrixXi& pp, 
	int ppow, double_vector_2d& data, int bin_size)
{

	int Nstart = 500; 
	
	double accept = 0, total = 0; 



	#pragma omp parallel 
	{

		thread_local std::random_device rd;

		uint32_t seed1, seed2; 
		#pragma omp critical
		{
			seed1 = rd(); 
			seed2 = rd(); 
		}

		thread_local std::mt19937 mt(seed2);
		thread_local std::mt19937 uni(seed1);

		std::uniform_int_distribution<> dist(0, Nsite-1);
		std::uniform_real_distribution<double> uniform_random(0, 1);

		MatrixXcd mv_prod(8*Nsite, Nhid);
		MatrixXcd new_mv_prod(8*Nsite, Nhid);

		VectorXcd all_coeff(8*Nsite); //save coeff for calculating energy
		VectorXcd new_all_coeff(8*Nsite);
		// VectorXcd tmp(8*Nsite);

		VectorXd new_state(Nsite); 
		VectorXd state(Nsite); 

		MatrixXd state_tran(8*Nsite, Nsite);
		MatrixXd new_state_tran(8*Nsite, Nsite);

		complex<double> coeff_new, coeff_old;

		bool initialized = false; 
		int id; 


		vector<double> st(Nsite, 0.5);
		for(int i=0;i<Nsite/2;i++)
			st[i] *= -1.0; 


		shuffle(st.begin(), st.end(), mt);

		// VectorXd state(Nsite); 
		for(int i=0;i<Nsite;i++)
			state[i] = st[i]; 


		for(int t=0;t<Nstart;t++)
		{
			int x = dist(mt);
			int y = x; 

			while(state[y] * state[x] > 0.0)
				y = dist(mt); 

			// VectorXcd temp(Nhid);
			new_state = state; 

			new_state[x] *= -1.0; 
			new_state[y] *= -1.0; 

			// MatrixXi tops(Nsite, 2);

			// cout<<"b1"<<endl; 

			if(!initialized)
			{
				translational_symmetry(state, state_tran); 
				coeff_old = coefficient(state_tran, W, mv_prod, all_coeff); 
				initialized = true;
			}
			

			// translational_symmetry(new_state, new_state_tran);

			new_state_tran = state_tran;

			coeff_new = complex<double>(0.0, 0.0);   //important

			for(int i=0;i<8*Nsite;i++)
			{
				complex<double> prod(1.0, 0); 
				id = i/Nsite;

				new_state_tran(i, pp(x, i)) *= -1.0;
				new_state_tran(i, pp(y, i)) *= -1.0;

				for(int j=0;j<Nhid;j++)
				{
					new_mv_prod(i, j) = mv_prod(i, j) - 2.0*( W(j, pp(x, i))*state_tran(i, pp(x, i)) +  W(j, pp(y, i))*state_tran(i, pp(y, i)) );
					// new_mv_prod(i, j) = mv_prod(i, j) - 2.0*( W(j, tops(i, 0))*state_tran(i, tops(i, 0)) +  W(j, tops(i, 1))*state_tran(i, tops(i, 1)) );
					prod *= cosh(new_mv_prod(i, j));
				} 

				// if(id == 2 or id == 3 or id == 4 or id == 6)
				// 	prod *= -1.0;

				new_all_coeff[i] = prod; 

				coeff_new += prod; 
			} 


			coeff_new /= (double)(Nsite*8);

			if(uniform_random(uni) < min(1.0, norm(coeff_new/coeff_old) ) )
			{
				state = new_state; 
				coeff_old = coeff_new;
				mv_prod = new_mv_prod;
				all_coeff = new_all_coeff;
				state_tran = new_state_tran;
			}	


		}



		// initialized = false;
		VectorXcd local_sum(10);
		local_sum *= 0.0; 

		double local_tot = 0, local_acc = 0;

		double_vector_2d local_data; 

		int tot_num = 0; 
	

		#pragma omp for
		for(int cnt=0;cnt<Nsample;cnt++)
		{
			
			// cout<<cnt<<endl; 

			for(int t=0;t<Nskip;t++)
			{

				int x = dist(mt);
				int y = x; 

				while(state[y] * state[x] > 0.0)
					y = dist(mt); 

				new_state = state;  

				new_state[x] *= -1.0; 
				new_state[y] *= -1.0; 

				
				if(!initialized)
				{
					translational_symmetry(state, state_tran); 
					coeff_old = coefficient(state_tran, W, mv_prod, all_coeff); 
					initialized = true;
				}


				// translational_position(x, y, Nsite, tops);
				// translational_symmetry(new_state, new_state_tran); 
				new_state_tran = state_tran;
				
				coeff_new = complex<double>(0.0, 0.0); //important

				for(int i=0;i<8*Nsite;i++)
				{
					complex<double> prod(1.0, 0); 
					id = i/Nsite;

					new_state_tran(i, pp(x, i)) *= -1.0;
					new_state_tran(i, pp(y, i)) *= -1.0;

					for(int j=0;j<Nhid;j++)
					{

						// if( pp(x,i) != tops(i, 0) or pp(y,i) != tops(i, 1) )
						// 	cout<<"wrong position"<<endl; 

						new_mv_prod(i, j) = mv_prod(i, j) - 2.0*( W(j, pp(x, i))*state_tran(i, pp(x, i)) +  W(j, pp(y, i))*state_tran(i, pp(y, i)) );
						// new_mv_prod(i, j) = mv_prod(i, j) - 2.0*( W(j, tops(i, 0))*state_tran(i, tops(i, 0)) +  W(j, tops(i, 1))*state_tran(i, tops(i, 1)) );
						prod *= cosh(new_mv_prod(i, j));
					}

					// if(id == 2 or id == 3 or id == 4 or id == 6)
					// 	prod *= -1.0;

					new_all_coeff[i] = prod; 
					coeff_new += prod; 
				} 

				coeff_new /= (double)(Nsite*8);

				local_tot += 1.0;
				if(uniform_random(uni) < min(1.0, norm(coeff_new/coeff_old) ) )
				{
					local_acc += 1.0;
					state = new_state; 
					coeff_old = coeff_new;
					mv_prod = new_mv_prod;
					all_coeff = new_all_coeff; 
					state_tran = new_state_tran;
				}	

			}
			
			
			tot_num ++; 
			complex<double> tmp_energy;
			complex<double> coeff = coeff_old; 
		
			// tmp_energy = Get_energy(state_tran, coeff, W,  Nhid, Nsite, J2, mv_prod, all_coeff); 
			tmp_energy = Get_energy_a1(state, state_tran, coeff, W, Nhid, Nsite, J2, mv_prod, pp);


			local_sum[0] += tmp_energy; 

			local_sum[1] += conj(tmp_energy)*tmp_energy;

			if(ppow != 0)
			{
				complex<double> hh = HHntr_a1(state, state_tran, W, Nhid, Nsite, J2, mv_prod, pp) / coeff; 
	
				local_sum[2] += conj(tmp_energy)*hh;
				local_sum[3] += conj(hh)*hh; 

				if(ppow == 2)
				{
					complex<double> hhh = HHHntr_a1(state, state_tran, W, Nhid, Nsite, J2, mv_prod, pp) / coeff;

					local_sum[4] += conj(hh)*hhh / (double)Nsample;

				}
			}



			if(tot_num % bin_size == 0)
			{
				tot_num = 0; 
				
				for(int c=0;c<5;c++)
					local_sum[c] /= (double)bin_size;  

				vector< complex<double> > vec{local_sum[0], local_sum[1]};

				if(ppow != 0)
				{
					vec.push_back(local_sum[2]); 
					vec.push_back(local_sum[3]);
					if(ppow == 2)
					{
						vec.push_back(local_sum[4]);
					}
				} 

				local_data.push_back(vec); 

				local_sum *= 0.0; 

			}
			

		}

		#pragma omp critical
		{

			data.insert(data.end(), local_data.begin(), local_data.end());

			accept += local_acc; 
			total += local_tot; 
		}


	}


	accept_rt = accept/total;

}













 

		
		



