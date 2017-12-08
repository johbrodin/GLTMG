
/*-----------------------------------
- GLT-MG; implementation in deal.ii 
- by Hreinn Juliusson aut.17
-------------------------------------*/

#ifndef mgPreconditionition_H
#define mgPreconditionition_H

#include <iostream>
#include <fstream>
#include <cmath>

#include <deal.II/base/config.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/identity_matrix.h> // IdentityMatrix inclusion :)

DEAL_II_NAMESPACE_OPEN
//using namespace mgPrecondition;

class mgPrecondition: public Subscriptor{

public:
	mgPrecondition(const SparseMatrix<double> &sparse_matrix,
		const Vector<double> &vector);
	void vmult(Vector<double> &dst, const Vector<double> &src) const;
	//GLTmg methods remove some for optimization
	void presmooth(Vector<double> &vicky);
	void postsmooth(Vector<double> &vicky);
	/* These method are used outside of this class */
	Vector<int> factor(const int N);
	Vector<int> unique(Vector<int> factor);
	Vector<int> accumVector(Vector<int> v);
	double vectorProd(Vector<int> v1, Vector<int> v2);
	void prol(double n, SparsityPattern &spP, SparseMatrix<double> &smP); // Change the return type ;)
	void kronProd(SparseMatrix<double> &A, SparseMatrix<double> &B,
		SparsityPattern &sp, SparseMatrix <double> &M);
	void kronProd_vector(Vector<double> &A, Vector<double> &B,
		SparsityPattern &sp, SparseMatrix <double> &M);
	void spdiags(double n,SparsityPattern &spaa, SparseMatrix <double> &aa);
	void transp(SparseMatrix<double> &A, SparsityPattern &sp, SparseMatrix<double> &M);
	// Help methods
	void printMatrix();
	void printVector();
	void sayHi();
private:
	const SmartPointer<const SparseMatrix<double>> system_matrix;
	const SmartPointer<const Vector<double>> rhs;
	SmartPointer<Vector<double>> test_vector;
	SmartPointer<SparseMatrix<double>> test_matrix;
        double tol;
};

mgPrecondition::mgPrecondition(const SparseMatrix<double> &sparse_matrix,
	const Vector<double> &vector):
system_matrix(&sparse_matrix),
rhs(&vector)
{
    tol = 0.00000001; //1e-8;
}

// This will be the GLTmg main function!
void mgPrecondition::vmult(Vector<double> &dst, const Vector<double> &src) const {
        //dst = src;

        dst=0;
        Vector<double> r(src.size());
        system_matrix->vmult(r,src);
        r -= src;

        std::cout<<" - - - - - - Ding - - - - - - -\n"<<std::endl;

        //TTTT

        /*% initial guess
        x = zeros(size(b));
        r = b-A*x;                % residual
        tol = tol*norm(b);

        i = 1;
        while norm(r) > tol && i <= nit
           % one step of MG-GLT
           x = mgGLT_setup(A, BB, PP, b, x, gamma, presmooth, postsmooth, n, 1);
           % new residual
           r = b-A*x;
           i = i+1;
        end
        iter = i-1;*/
}

/* ===================== The functions needed to perform GLTmg ==============================================================*/

/* Presmoot alg.
	diagA = diag(A);
	x = x+1*(b-A*x).diagA;
	We use the predefined Jacobi_step
*/
void mgPrecondition::presmooth(Vector<double> &vicky){
	system_matrix->Jacobi_step(vicky,*rhs,1); // Remember smartpointer to rhs!
}

void mgPrecondition::postsmooth(Vector<double> &vicky){
	const double damp = 2./3;
	system_matrix->Jacobi_step(vicky,*rhs,damp);
}

/* Return a vector containing the primenumbers of N 
	Might Implement Sieve at a later point 		*/
Vector<int> mgPrecondition::factor(int N){
	int index = 0;
	const int maxSize = 10;
	int factors[maxSize];
	while(N%2==0){
		//Add 2
		factors[index] = 2;
		index++;
		N=N/2;
	}
	for(int i=3; i<=sqrt(N); i=i+2){
		while(N%i==0){
			//add i
			factors[index] = i;
			index++;
			N=N/i;
		}
	}
	if(N>2){
		//add (N)
		factors[index] = N;
		index++;
	}
	Vector<int> result;
	result.reinit(index);
	Vector<int>::iterator iter = result.begin();
	Vector<int>::iterator ender = result.end();
	int i = 0;
	for(; iter!=ender; iter++){
		*iter = factors[i];
		i++;
	}
	return result;
}

/* removes extra numbers from factors */
Vector<int> mgPrecondition::unique(Vector<int> factor){
	if(factor.size()==1){
		return factor;
	} else{
		Vector<int> unique;
		const int maxSize = 10;
		int tmp[maxSize];
		int i = 0;
		Vector<int>::iterator iter = factor.begin();
		Vector<int>::iterator ender = factor.end();
		tmp[i] = *iter;
		iter++;
		for(; iter!=ender; iter++){
			if(*iter!=tmp[i]){
				i++;
				tmp[i] = *iter;
			}
		}
		unique.reinit(i+1);
		iter = unique.begin();
		ender = unique.end();
		for(i=0;iter!=ender; iter++){
			*iter = tmp[i];
			i++;
		}
		return unique;
	}
}

/* Returns an accumVector. This method assumes v is sorted! 
	Note! The GLTmg has *ugh* Matlab indexing !
	Returnes # of Values of pos int on index i !
*/
Vector<int> mgPrecondition::accumVector(Vector<int> v){
	Vector<int>::iterator iter = v.begin();
	Vector<int>::iterator ender = v.end();
	Vector<int> res;
	int res_size = *(ender-1);
	res.reinit(res_size); //prolong so you can access the last element!
	res = 0;
	for(; iter!=ender; iter++){
		res(*iter-1)++;
	}
	return res;
}

/* This method performs the prod(nu.^(resp(nu)./3)) 
	action given in the GLTmg*/
double mgPrecondition::vectorProd(Vector<int> reps, Vector<int> nu){
	double n = 1;
	// the reps(nu) action
	Vector<double> repsNu;	//Used for development
	repsNu.reinit(nu.size());
	Vector<int>::iterator nu_iter = nu.begin();
	Vector<int>::iterator nu_ender = nu.end();
	int i =  0;
	for(; nu_iter!=nu_ender; nu_iter++){
		repsNu(i) = (double)(reps(*nu_iter-1))/3;
		n = n*pow(nu(i),repsNu(i));
		i++;
	}
	return n;
}
/*
	This is the kronecker product of 2 sparse matrices
*/
void mgPrecondition::kronProd(SparseMatrix<double> &A, SparseMatrix<double> &B,
	SparsityPattern &sp, SparseMatrix <double> &M){
	const int nA = A.n(); //nCols
	const int nB = B.n();
	const int mA = A.m(); //nRows
	const int mB = B.m();
	const int n = nA*nB;
	const int m = mA*mB;
	DynamicSparsityPattern dsp(m,n);
	SparseMatrix<double>::iterator itA = A.begin();
	SparseMatrix<double>::iterator endA = A.end();
	SparseMatrix<double>::iterator itB = B.begin();	// I fixed it ;P 
	SparseMatrix<double>::iterator endB = B.end();
	int rowA, colA, rowB, colB;
	int i,j;
	double val,valA,valB;
	for(; itA!=endA; itA++){
		rowA = itA->row();
		colA = itA-> column();
		for(; itB!=endB; itB++){
			rowB = itB->row();
			colB = itB-> column();
			i = rowA*mB+rowB;
			j = colA*nB+colB;
			dsp.add(i,j);
		}
		itB = B.begin();
	}
	dsp.compress();
	// Should we put a sp.clear?
	sp.copy_from(dsp);
	M.reinit(sp);
	itA = A.begin();
	for(; itA!=endA; itA++){
		rowA = itA->row();
		colA = itA-> column();
		valA = itA->value();
		for(; itB!=endB; itB++){
			rowB = itB->row();
			colB = itB-> column();
			valB = itB->value();
			i = rowA*mB+rowB;
			j = colA*nB+colB;
			val = valA*valB;
			M.add(i,j,val);
		}
		itB = B.begin();
	}
}

/* Vector A is percieved to be a 1xn vector
and Vector B is percieved to be a mx1 vector!
	resulting Matrix M is a mxn matrix
*/
void mgPrecondition::kronProd_vector(Vector<double> &A, Vector<double> &B,
	SparsityPattern &sp, SparseMatrix <double> &M){
	const int m = B.size();
	const int n = A.size();
	DynamicSparsityPattern dsp(m,n);
	Vector<double>::iterator itA = A.begin();
	Vector<double>::iterator itB = B.begin();
	Vector<double>::iterator endA = A.end();
	Vector<double>::iterator endB = B.end();
	int i=0,j=0;
	double val,valA,valB;
	for(; itA!=endA; itA++){
		for(; itB!=endB; itB++){
			dsp.add(j,i);
			j++;
		}
		itB = B.begin();
		j=0;
		i++;
	}
	dsp.compress();
	sp.copy_from(dsp);
	M.reinit(sp);
	itA = A.begin();
	i=0; j=0;
	for(; itA!=endA; itA++){
		valA = *itA;
		for(; itB!=endB; itB++){
			valB = *itB;
			val = valA*valB;
			M.add(j,i,val);
			j++;
		}
		itB = B.begin();
		j=0;
		i++;
	}
}

/* 	Here we make a transpose of matrix A which can be any type of a matrix  */
void mgPrecondition::transp(SparseMatrix<double> &A, SparsityPattern &sp, SparseMatrix<double> &M){
	const int n = A.m();
	const int m = A.n();
	DynamicSparsityPattern dsp(m,n);
	SparseMatrix<double>::iterator itA = A.begin();
	SparseMatrix<double>::iterator endA = A.end();
	double value;
	int rowM=0,colM=0;
        for(;itA!=endA;++itA){
		rowM=itA->column();
		colM=itA->row();
		dsp.add(rowM,colM);
	}
	dsp.compress();
	sp.copy_from(dsp);
	M.reinit(sp);
	itA = A.begin();
	for(;itA!=endA;itA++){
		rowM=itA->column();
		colM=itA->row();
		value = itA->value();
		M.add(rowM,colM,value);
	}
}


/* Here we implement the prol([2 1 1],[0 -1 1],n)
	dd = [0 -1 1]
	PP = [2 1 1]
	performs the operation:
	aa = spdiags(kron(PP,one(n,1),dd,n,n));
*/
void mgPrecondition::spdiags(double n,SparsityPattern &spaa, SparseMatrix <double> &aa){
	Vector<double> dd;
	dd.reinit(3);
	dd[0] = 0; dd[1] = -1; dd[2] = 1;
	Vector<double> PP;
	PP.reinit(3);
	PP[0] = 2; PP[1] = 1; PP[2] = 1; // Obs!!!
	int N = (int)n;
	/* Make the sm1 = kron(PP,ones(n,1)) matrix! :D */
	Vector<double> ones;
	ones.reinit(N);
	ones = 1;
	SparsityPattern sp1;
	SparseMatrix<double> sm1;
	kronProd_vector(PP,ones,sp1,sm1);
	/* I am hardcoding the diagonals. i.e. I dont use the dd matrix. Sorry ;)*/
	DynamicSparsityPattern dsp(N,N);
	// ! ! ! ! !  Possible problem if the matrix is shorter than N !! ! ! !
	for(int i=0; i<N-1; i++){
		dsp.add(i,i);
		dsp.add(i+1,i);
		dsp.add(i,i+1);
	}
	dsp.add(N-1,N-1);
	dsp.compress();
	spaa.copy_from(dsp);
	aa.reinit(spaa);
	for(int i=0; i<N-1; i++){
		aa.add(i,i,sm1(i,0));		// Column 0 on diag 0
		aa.add(i+1,i,sm1(i+1,1));	// Column 1 on diag -1
		aa.add(i,i+1,sm1(i,2));		// Column 2 on diag 1
	}
	aa.add(N-1,N-1,sm1(N-1,0));
}

/* This is an implementation of the algorithm 
	(1/n)*prol([2 1 1], [0 -1 1],n)
	given in the gltmg_test matlab code 
	*/
void mgPrecondition::prol(double n, SparsityPattern &spP, SparseMatrix<double> &P){
	// aa = spdiags(kron(PP,ones),dd,n,n)
	SparsityPattern spaa;
	SparseMatrix<double> aa;
	spdiags(n,spaa,aa);
	// smP = kron(aa,kron(aa,aa))
	SparsityPattern spTemp;
	SparseMatrix<double> smTemp;
	kronProd(aa,aa,spTemp,smTemp);
	SparseMatrix<double> smP;
	SparsityPattern spP2;
	kronProd(aa,smTemp,spP2,smP);
	/* Create smH = speye(n) then remove every other row
		smH = smH(1:2:n,:)*/
	SparsityPattern spH2;
	SparseMatrix<double> smH;
	int N1 = (int)n;
	int N2 = N1/2+N1%2;
	DynamicSparsityPattern dspH(N2,N1);
	for(int i=0; i<N2; i++){
		dspH.add(i,2*i);
	}
	dspH.compress();
	spH2.copy_from(dspH);
	smH.reinit(spH2);
	for(int i=0; i<N2; i++){
		smH.add(i,2*i,1);
	}
	smH.print_formatted(std::cout,1,true,0," ",1);
	// H = kron(smH,smH)
	SparsityPattern spH;
	SparseMatrix<double> H;
	kronProd(smH,smH,spH,H);
	// P = smP*H'; P = (1/n)*P
	std::cout<<" - - - -- - - "<<std::endl;
	H.print_formatted(std::cout,1,true,0," ",1);
	SparseMatrix<double> transpH;
	SparsityPattern spTranspH;
	transp(H,spTranspH,transpH);
	DynamicSparsityPattern dspP(0);
	spP.copy_from(dspP);
	P.reinit(spP);
	smP.mmult(P,transpH,Vector<double>(),true);

}


/*==========================================================================================================================*/

/* Test/Help functions! */
void mgPrecondition::sayHi(){
	std::cout<<" - - - - Hi There! - - - -  "
	<<std::endl;
}
void mgPrecondition::printMatrix(){
	//Why do we use arrow here? -> = (*).
        system_matrix->print_formatted(std::cout,1,true,0," ",1);
}
void mgPrecondition::printVector(){
	rhs->print(std::cout,1,true,true);	
}

void foo(int &N){
	N+=30;
}


DEAL_II_NAMESPACE_CLOSE
#endif



/*

 __
 | |
 | |
 | |
 | |
 | |		   _____
 | |		  |		|	  |
 | | Hreinn	  |		|	  |
_| |__________|		|_____|_________________
*/
