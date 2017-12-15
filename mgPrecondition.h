
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
#include <vector> // This is for the std::vector

DEAL_II_NAMESPACE_OPEN
//using namespace mgPrecondition;

class mgPrecondition: public Subscriptor{

public:
	mgPrecondition(SparseMatrix<double> *bb, SparseMatrix<double> *pp,
   Vector<double> &vector);
	void vmult(Vector<double> &dst, const Vector<double> &src) const;
	//GLTmg methods remove some for optimization
	void presmooth(Vector<double> &vicky);
	void postsmooth(Vector<double> &vicky);
	// Help methods
    void testPP();
    void printMatrix();
    void printVector();
    void sayHi();
    //Johanna:
    void mgRecursion(Vector<double> &dst_x, const Vector<double> &src_b, int level) const; //const to be able to be called by the const vmult function
    void newResidual(Vector<double> &r,Vector<double> &x,const Vector<double> &b,SparseMatrix<double> &A) const;// )const;//
    void presmooth_test(Vector<double> &dst, const Vector<double> &src, const SparseMatrix<double> *&A) const;
    //void mgRecursion(Vector<double> &dst_x, const Vector<double> &src_x, int level) const; //const to be able to be called by the const vmult function
    //void newResidual(Vector<double> &r,Vector<double> &x,const Vector<double> &b,SparseMatrix<double> &A) const;// )const;//
    //void presmooth_test(Vector<double> &dst, Vector<double> &src, const SparseMatrix<double> *&A);
private:
    SparseMatrix<double> *BB;
    SparseMatrix<double> *PP;
    Vector<double> rhs;
        //Johanna:
    double tol; //tolerance
    double max_iterations; //max number of MG cycles

        // Alternative BB and PP
};

mgPrecondition::mgPrecondition(SparseMatrix<double> *bb, SparseMatrix<double> *pp,
   Vector<double> &vector){
    tol = 0.00000001; //1e-8;
    max_iterations = 3; //number of MG cycles
    PP = pp;
    BB = bb;
    rhs = vector;
}

/* ===================== The functions needed to perform GLTmg ==============================================================*/
/* This will be the GLTmg main function!
 * */
void mgPrecondition::vmult(Vector<double> &dst, const Vector<double> &src) const {
    //dst = src; x = b
    //std::cout<<" - - - - - - Ding - - - - - - -\n"<<std::endl;
    dst=0;  //initial guess
    Vector<double> r(src.size());
  //  newResidual(r,dst,src,BB[0]);//);//
    double tol_res = tol*src.l2_norm();

    int level = 0;
    while (r.l2_norm() > tol_res && level <= max_iterations)
    {
    	mgRecursion(dst,src,level);
    }
    
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

/* ===================== Help functions for vmult (/Johanna) ==============================================================*/
//can probably make better, but is tested, works
void mgPrecondition::newResidual(Vector<double> &r,Vector<double> &x,const Vector<double> &b,SparseMatrix<double> &A) const {// )const{//
    //residual  r=b-A*x
        A.vmult(r,x);    //, r_temp=A*x
        r -= b;// r=r_temp-b, but *-1 needed to get r=b-A*x
        double sign = -1.0;
        r*=sign;
}
void mgPrecondition::mgRecursion(Vector<double> &dst_x, const Vector<double> &src_b, int level) const {
    /*% at the last level the system is solved directly
    if (n < 5)
       x = A\b;
    else*/

    //const SparseMatrix<double>* P(PP[level]);
   // presmooth_test(dst_x,src_b,BB[0]);
    Vector<double> r(src_b.size());
    //newResidual(r,dst_x,src_b,*BB[0]);

    /*P =PP{liv};% projection matrix at the current level
    x = presmooth(A,b,x);% v1 steps of the pre-smoother
    r = b-A*x;% residual at the finer grid
    d = P'*r; % restriction of the residual to the coarser grid

       % dimension of the problem at the coarser level
    %    if (mod(n,2) == 0)
    %        k = n/2;
    %    else
    %        % k=(n-1)/2;   %%%%% n=2^t-1 %%%%%%
    %        k = (n+1)/2;  %%%%% n=2^t+1 %%%%%%
    %
    %    end
       k = floor((n+1)*0.5);
    %    e = zeros(k^3,1);
       e = zeros(size(d));% the initial error at the coarse grid is zero
       for j=1 : gamma   % recursive call of the MG-GLT
    %       B = P'*A*P;
          B = BB{liv+1};
          e = mgGLT_setup(B, BB, PP, d, e, gamma, presmooth, postsmooth, k, liv+1);
       end
       g = P*e;% the error e is interpolated back to obtain the finer level error
       x = x + g;% updating of the solution with the error at the finer level
       x = postsmooth(A,b,x);% v2 steps of post-smoother*/
}
void mgPrecondition::presmooth_test(Vector<double> &dst,const Vector<double> &src,const SparseMatrix<double> *&A) const{
        A->Jacobi_step(dst,src,1);
}

/* ===================== The functions needed to perform GLTmg ==============================================================*/

/* Presmoot alg.
        diagA = diag(A);
        x = x+1*(b-A*x).diagA;
        We use the predefined Jacobi_step
*/
void mgPrecondition::presmooth(Vector<double> &vicky){
    //system_matrix->Jacobi_step(vicky,*rhs,1); // Remember smartpointer to rhs!
}

void mgPrecondition::postsmooth(Vector<double> &vicky){
    const double damp = 2./3;
    //system_matrix->Jacobi_step(vicky,*rhs,damp);
}

void mgPrecondition::testPP(){
  std::cout<<" PP[0] mxn : "<<PP[0].m()<<" "<<PP[0].n()<<std::endl;
  std::cout<<" PP[1] mxn : "<<PP[1].m()<<" "<<PP[1].n()<<std::endl;
  std::cout<<" BB[0] mxn : "<<BB[0].m()<<" "<<BB[0].n()<<std::endl;
  std::cout<<" BB[1] mxn : "<<BB[1].m()<<" "<<BB[1].n()<<std::endl;
  std::cout<<" BB[2] mxn : "<<BB[2].m()<<" "<<BB[2].n()<<std::endl;
}

/*==========================================================================================================================*/

/* Test/Help functions! */
void mgPrecondition::sayHi(){
    std::cout<<" - - - - Hi There! - - - -  "
    <<std::endl;
}
void mgPrecondition::printMatrix(){
	//Why do we use arrow here? -> = (*).
	//system_matrix->print_formatted(std::cout,1,true,0," ",1);
}
void mgPrecondition::printVector(){
    //rhs->print(std::cout,1,true,true);
}

void foo(int &N){
    N+=30;
}


DEAL_II_NAMESPACE_CLOSE
#endif



/*

 __
 | |    The code dump
 | |
 | |
 | |
 | |		   _____
 | |		  |		|	   _____
 | | Hreinn	  |		|	  |     |
_| |__________|		|_____|     |___________


    // PP{i} = 0 gets overwritten in the loop :'(
    //SparsityPattern spPoint;
    //const SparseMatrix<double> point = pointMatrix(spPoint);
    //PP.push_back(&point);



*/
