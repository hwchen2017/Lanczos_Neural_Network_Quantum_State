


#include "nat_grad.h"


void mat_vec_mult(complex<double>* & pDq, complex<double>* & pdq, complex<double>* & px,  complex<double>* & pAx,int N, int T, double lam){

	

		complex<double>* py =(std::complex<double> *)mkl_malloc(T*sizeof( std::complex<double> ), 64 );
		complex<double> lamc=complex<double>(lam,0.0);
		const complex<double> c1=complex<double>(1.0,0.0);
		const complex<double>c2=complex<double>(0.0,0.0);
		complex<double> ol;


		cblas_zgemv(CblasColMajor,CblasConjTrans,
		N,T,&c1,pDq,N,px,1,&c2,py,1);
		cblas_zgemv(CblasColMajor,CblasNoTrans,
		N,T,&c1,pDq,N,py,1,&c2,pAx,1);

		cblas_zdotc_sub(N,pdq,1,px,1,&ol);
		ol=-ol;
		
		cblas_zaxpy(N,&ol,pdq,1,pAx,1);
		cblas_zaxpy(N,&lamc,px,1,pAx,1);
		
		mkl_free(py);


}











void Nat_Grad_CG(complex<double>* pDq, complex<double>* pdq, complex<double>* pg, complex<double>* pg2, double & tol, int N, int T, int max_it, double lam) {
	
	bool stop = false;
	int it = -1;

	//nt = 1;
	


	double tol_stop=tol;
	
	double tolbest = 100.0;
	int itbest = 0;

	//mkl_domain_set_num_threads(nt);

	
	
	complex<double> *pg2b,*pr,*pp,*pAp;
	
	pg2b =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64 );
	pr =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64);
	pp =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64 );
	pAp =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64 );
	//py =(std::complex<double> *)mkl_malloc(T*sizeof( std::complex<double> ), 64 );

	
	cblas_zcopy(N,pg2,1,pr,1);
	
	//double dsum = cblas_dzasum(N,pDqd,1);
	//dsum = dsum/(double)(N);
	//lam*=dsum;


	double diag_avg=0.0;
	diag_avg=cblas_dznrm2(T*N,pDq,1);
	diag_avg*=diag_avg;
	double dqnorm=cblas_dznrm2(N,pdq,1);
	diag_avg=diag_avg-dqnorm*dqnorm;
	diag_avg*=(1.0/(double)(N));
	lam*=diag_avg;


	complex<double> alpha,c1,c2;
	c1=complex<double>(1.0,0.0);
	c2=complex<double>(0.0,0.0);

	mat_vec_mult(pDq,pdq, pg2,  pr,N, T,lam);
	complex<double> q0;
	cblas_zdotc_sub(N,pg2,1,pr,1,&q0);
	complex<double> q1;
	cblas_zdotc_sub(N,pg2,1,pg,1,&q1);
	//q0=0.0;
	q1=complex<double>(0.0,0.0);
	if(abs(q1/q0) >.01){
		q1=(q1/q0);
		cblas_zscal(N,&q1,pg2,1);
		q1=-q1;
		cblas_zaxpby(N,&c1,pg,1,&q1,pr,1);
		
	}else{
		q1=complex<double>(0.0,0.0);
		cblas_zscal(N,&q1,pg2,1);
		cblas_zcopy(N,pg,1,pr,1);
	}

	
	cblas_zcopy(N,pr,1,pp,1);
	cblas_zcopy(N,pg2,1,pg2b,1);


	double gnorm,rsnew,rsold,beta;

	gnorm = cblas_dznrm2(N,pg,1);
	//cout << "gnorm: " << gnorm << endl;
	//gnorm= gnorm*gnorm;

	rsold = cblas_dznrm2(N,pr,1);
	rsold = rsold*rsold;

	

	//omp_set_num_threads(omp_get_num_procs());
	//omp_set_num_threads(nt);
	
	while (!stop) {
		it = it + 1;
		
		
		
		mat_vec_mult(pDq,pdq, pp,  pAp,N, T,lam);
		
		cblas_zdotc_sub(N,pp,1,pAp,1,&alpha);
		alpha = rsold / alpha;
		


		
		cblas_zaxpy(N,&alpha,pp,1,pg2,1);
		alpha=-alpha;
		cblas_zaxpy(N,&alpha,pAp,1,pr,1);

		tol = cblas_dznrm2(N,pr,1);
		rsnew=tol*tol;
		tol=(tol/gnorm);

		//tol=cblas_dznrm2(N,pr,1);
		//cout << it << " " << tol << endl;
		//tol=(tol/gnorm);
		//cout << it << " " << tol << endl;

		if ((tol < tol_stop) || (it > max_it)) {
			stop = true;
			if (tol < tolbest) {
				cblas_zcopy(N,pg2,1,pg2b,1);
				tolbest = tol;
				itbest = it;
			}
		}
		else {
			beta = (rsnew / rsold);
			cblas_zdscal(N,beta,pp,1);
			cblas_zaxpy(N,&c1,pr,1,pp,1);
			rsold = rsnew;
			if (tol < tolbest) {
				cblas_zcopy(N,pg2,1,pg2b,1);
				tolbest = tol;
				itbest = it;
			}

		}

		


	}
	//cout << tol << " " << tolbest << endl;
	tol=tolbest;
	cblas_zcopy(N,pg2b,1,pg2,1);

	mkl_free(pg2b);
	mkl_free(pr);
	mkl_free(pp);
	mkl_free(pAp);
	//mkl_free(py);


	//cout << g2 << endl;
	//cout << tol << endl;
	//cout << it << endl;
	
	

	
	//return g2;

}


void Nat_Grad_CG_mth(complex<double>* pDq, complex<double>* pdq, complex<double>* pg, complex<double>* pg2, double & tol, int N, int T, int max_it,double lam) {

	//double time_init = omp_get_wtime();
	//double time_real_tot =omp_get_wtime();
	bool stop = false;
	int it = 0;
	double tol_stop=tol;
	
	double tolbest = 100.0;
	int itbest = 0;


	double diag_avg=0.0;
	diag_avg=cblas_dznrm2(T*N,pDq,1);
	diag_avg*=diag_avg;
	double dqnorm=cblas_dznrm2(N,pdq,1);
	diag_avg=diag_avg-dqnorm*dqnorm;
	diag_avg*=(1.0/(double)(N));
	lam*=diag_avg;






	complex<double> *pg2b,*pr,*pp,*pAp,*py;
	
	pg2b =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64 );
	pr =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64);
	pp =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64 );
	pAp =(std::complex<double> *)mkl_malloc(N*sizeof( std::complex<double> ), 64 );
	py =(std::complex<double> *)mkl_malloc(T*sizeof( std::complex<double> ), 64 );

	cblas_zcopy(N,pg,1,pr,1);
	cblas_zcopy(N,pg,1,pp,1);


	complex<double> alpha,c1,c0,ol;
	c1=complex<double>(1.0,0.0);
	c0=complex<double>(0.0,0.0);


	cblas_zscal(N,&c0,pg2,1);
	cblas_zcopy(N,pg2,1,pg2b,1);
	double gnorm,rsnew,rsold,beta;
	gnorm = cblas_dznrm2(N,pg,1);
	rsold = cblas_dznrm2(N,pr,1);
	rsold = rsold*rsold;

	//time_init = omp_get_wtime()-time_init;
	//cout << "init time: " << time_init << endl;

	//omp_set_num_threads(omp_get_num_procs());
	//omp_set_num_threads(nt);

	#pragma omp parallel
	{



		//double time0=0.0,time1=0.0,time2=0.0,time3=0.0,time4=0.0,time_tot=omp_get_wtime();
		//double dt=omp_get_wtime();

		int ntc = omp_get_num_threads();
		
		int ID = omp_get_thread_num();
		int cn = T / ntc;
		int rn = T % ntc;

		int i0 = cn * ID;
		int i1 = cn * (ID + 1);
		if (ID < rn) {
			i0 = i0 + ID;
			i1 = i1 + ID + 1;

		}
		else {
			i0 += rn;
			i1 += rn;
		}
		int Tc=(i1-i0);


		cn = N / ntc;rn = N % ntc;
		
		int j0 = cn * ID;
		int j1 = cn * (ID + 1);
		if (ID < rn) {
			j0 = j0 + ID;
			j1 = j1 + ID + 1;

		}
		else {
			j0 += rn;
			j1 += rn;
		}
		int Nc=(j1-j0);

		const complex<double> z1=complex<double>(1.0,0.0);
		const complex<double> z0=complex<double>(0.0,0.0);

		complex<double>* pDqc=(std::complex<double> *)mkl_malloc(Nc*T*sizeof( std::complex<double> ), 64 );
		complex<double>* pDqhc= (std::complex<double> *)mkl_malloc(N*Tc*sizeof( std::complex<double> ), 64 );


		mkl_zomatcopy('C','N',Nc,T,z1,pDq+j0,N,pDqc,Nc);
		mkl_zomatcopy('R','R',Tc,N,z1,pDq+i0*N,N,pDqhc,N);

		complex<double>* ppc = pp+j0;
		complex<double>* pyc = py+i0;
		complex<double>* pApc = pAp+j0;

		//dt=omp_get_wtime()-dt;
		
		//time0+=dt;

	#pragma omp barrier
	
	while (!stop) {
		/*
		#pragma omp critical
		{ cout << "thread: " << ID << " starting iteration " << it << endl;}
		*/
		//dt=omp_get_wtime();

		cblas_zgemv(CblasRowMajor,CblasNoTrans,Tc,N,&z1,pDqhc,N,pp,1,&z0,pyc,1);
		cblas_zcopy(Nc,ppc,1,pApc,1);
		cblas_zdscal(Nc,lam,pApc,1);
		#pragma omp single
		{
			//cout << "thread: " << ID << " doing single 1" << endl;
			cblas_zdotc_sub(N,pdq,1,pp,1,&ol);
			ol=-ol;
		}

		//dt=omp_get_wtime()-dt;
		//time1+=dt;
		/*
		#pragma omp critical
		{ cout << "thread: " << ID << " starting part 2" << endl;}
*/

		//dt=omp_get_wtime();

		cblas_zaxpy(Nc,&ol,pdq+j0,1,pApc,1);
		cblas_zgemv(CblasColMajor,CblasNoTrans,Nc,T,&z1,pDqc,Nc,py,1,&z1,pApc,1);
#pragma omp barrier
		#pragma omp single
		{
		
			//cout << "thread: " << ID << " doing single 2" << endl;
			cblas_zdotc_sub(N,pp,1,pAp,1,&alpha);
			alpha = rsold / alpha;

		}
		/*
		#pragma omp critical 
		{ cout << "thread: " << ID << " starting part 3" << endl;}
*/

		//dt=omp_get_wtime()-dt;
		//time2+=dt;

		//dt=omp_get_wtime();

		complex<double> alphac=alpha;		
		cblas_zaxpy(Nc,&alphac,ppc,1,pg2+j0,1);
		alphac=-alphac;
		cblas_zaxpy(Nc,&alphac,pApc,1,pr+j0,1);

		#pragma omp barrier
		//dt=omp_get_wtime()-dt;
		//time3+=dt;

		//dt=omp_get_wtime();
		#pragma omp single
		{

			//cout << "thread: " << ID << " doing single 3" << endl;
			++it;
			tol = cblas_dznrm2(N,pr,1);
			rsnew=tol*tol;
			tol=(tol/gnorm);

	
			if ((tol < tol_stop) || (it > max_it)) {
				stop = true;
				if (tol < tolbest) {
					cblas_zcopy(N,pg2,1,pg2b,1);
					tolbest = tol;
					itbest = it;
				}
			}
			else {
				beta = (rsnew / rsold);
				cblas_zdscal(N,beta,pp,1);
				cblas_zaxpy(N,&z1,pr,1,pp,1);
				rsold = rsnew;
				if (tol < tolbest) {
					cblas_zcopy(N,pg2,1,pg2b,1);
					tolbest = tol;
					itbest = it;
				}

			}

		}

/*
		#pragma omp critical
		{ cout << "thread: " << ID << " ending iteration " << it-1 << endl;}
*/
		#pragma omp barrier
		//dt=omp_get_wtime()-dt;
		//time4+=dt;
		


	}

		mkl_free(pDqc);mkl_free(pDqhc);


		//time_tot=time0+time1+time2+time3+time4;
		//time_tot=omp_get_wtime()-time_tot;
/*
		#pragma omp critical
		{

			cout << "thread: " << ID << "time_tot: " << time_tot << " t0: " << time0/time_tot << " t1: " << time1/time_tot << " t2: " << time2/time_tot << " t3: " << time3/time_tot << 
" t4: " << time4/time_tot << " rem: " << (time_tot-time0-time1-time2-time3-time4)/time_tot << endl;
		}


*/


	}
	//cout << tol << " " << tolbest << endl;
	tol=tolbest;
	cblas_zcopy(N,pg2b,1,pg2,1);

	mkl_free(pg2b);
	mkl_free(pr);
	mkl_free(pp);
	mkl_free(pAp);
	mkl_free(py);


	//cout << g2 << endl;
	//cout << tol << endl;
	//cout << it << endl;
	
	

	
	//return g2;

	//time_real_tot =omp_get_wtime()-time_real_tot;
	//cout << "real tot time: " << time_real_tot << endl;



}
