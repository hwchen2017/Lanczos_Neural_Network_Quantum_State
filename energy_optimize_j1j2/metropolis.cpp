#include <iostream>
#include <ctime>
#include "metropolis.h"
// #include "stoch_reconfig.h"
#include "nat_grad.h"
#include "functions.h"


complex<double> metropolis(MatrixXcd& W,  double lamda, int Nhid, int Nsite, int Nsample, double J2, int Nskip, double& accept_rt, double& rtol, MatrixXi& pp)
{

	int Nstart = 100; 

	complex<double> sum_energy(0, 0);
	
	Rmatrixxcd sum_partial = Rmatrixxcd::Zero(Nhid, Nsite); 
	MatrixXcd sum_HO = MatrixXcd::Zero(Nhid, Nsite);
	Rmatrixxcd derivative(Nhid, Nsite); 

	MatrixXcd A(Nhid*Nsite, Nsample);

	
	double total = 0.0, accept = 0.0; 
	


	// if(nt != 0) omp_set_num_threads(nt);

	// cout<<"bo"<<endl;

	// omp_set_num_threads(4);
	// double time1 = omp_get_wtime();

	#pragma omp parallel
	{	

		thread_local std::random_device rd;
		thread_local std::mt19937 mt(rd());
		thread_local std::mt19937 uni(rd());

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

				new_state_tran(i, pp(x, i)) *= -1.0;
				new_state_tran(i, pp(y, i)) *= -1.0;

				for(int j=0;j<Nhid;j++)
				{
					new_mv_prod(i, j) = mv_prod(i, j) - 2.0*( W(j, pp(x, i))*state_tran(i, pp(x, i)) +  W(j, pp(y, i))*state_tran(i, pp(y, i)) );
					// new_mv_prod(i, j) = mv_prod(i, j) - 2.0*( W(j, tops(i, 0))*state_tran(i, tops(i, 0)) +  W(j, tops(i, 1))*state_tran(i, tops(i, 1)) );
					prod *= cosh(new_mv_prod(i, j));
				} 

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

		complex<double> local_sum_energy(0, 0);
		double local_acc = 0.0, local_tot = 0.0; 
	
		Rmatrixxcd local_sum_partial = Rmatrixxcd::Zero(Nhid, Nsite); 
		MatrixXcd local_sum_HO = MatrixXcd::Zero(Nhid, Nsite);

		#pragma omp for
		for(int cnt=0;cnt<Nsample;cnt++)
		{
			
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
		


			complex<double> tmp_energy;
			Rmatrixxcd tmp_partial = Rmatrixxcd::Zero(Nhid, Nsite);  //must zero

			complex<double> coeff = coeff_old;
			
			// tmp_energy = Get_energy(state_tran, coeff, W,  Nhid, Nsite, J2, mv_prod, all_coeff);
			
			tmp_energy = Get_energy_a1(state, state_tran, coeff, W, Nhid, Nsite, J2, mv_prod, pp);
			coeff_partial(state_tran, W, tmp_partial, Nhid, mv_prod, all_coeff);

			// if(isnan(real(tmp_energy)))
			// 	cout<<coeff<<endl; 
			// cout<<tmp_energy<<endl; 
			
		 
			local_sum_energy += tmp_energy; 
			tmp_partial = tmp_partial.conjugate();

			local_sum_partial += tmp_partial; 
			local_sum_HO += tmp_partial * tmp_energy; 

			A.col(cnt) = Map<VectorXcd> (tmp_partial.data(), tmp_partial.size()); 
		}

		#pragma omp critical
		{
			sum_energy += local_sum_energy; 
			sum_partial += local_sum_partial; 
			sum_HO += local_sum_HO; 
			accept += local_acc; 
			total += local_tot;
		}


	}


	// double time2 = omp_get_wtime();

	// cout<<"sampling time: "<<time2 - time1<<"s"<<endl; 
	

	accept_rt = accept / total;

	sum_energy /= (double)Nsample; 
	sum_partial /= (double)Nsample; 
	sum_HO /= (double)Nsample; 
	derivative = sum_HO *2.0 - sum_partial*sum_energy*2.0; 

	Map<VectorXcd> grad(derivative.data(), derivative.size()); 
	Map<VectorXcd> der_avg(sum_partial.data(), sum_partial.size()); 
	VectorXcd x = VectorXcd::Zero(Nhid*Nsite);

	A /= (double)sqrt(Nsample); 

	complex<double> *pA = A.data();
	complex<double> *pder = der_avg.data(); 
	complex<double> *pgrad = grad.data(); 
	complex<double> *px = x.data(); 
	double tol = pow(10,-3);

	Nat_Grad_CG(pA, pder, pgrad, px, tol, Nhid*Nsite, Nsample, 100, pow(10,-6));
	

	Map<Rmatrixxcd> gradient(x.data(), Nhid, Nsite); 

	// cout<<gradient<<endl; 

	W -= lamda * gradient; 
	rtol = tol;

	// double time3 = omp_get_wtime();

	// cout<<"sampling time: "<<time3 - time2<<"s"<<endl; 


	return sum_energy; 

}



		
		



