/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
  * ---------------------------------------------------------------------
 * 
 *  Modified by the G8 group UU 2017
 * 
 * 
 */
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/tensor.h> // Tensor
#include <vector> // This is for the std::vector
#include <cmath>
//#include <deal.II/multigrid/mg_matrix.h> // For the mg::Matrix ;)

//#include "helloHeader.h"
//#include "helloPrecondition.h"
//#include "glt_init.h"
#include "mgPrecondition.h"

using namespace dealii;
class Step3
{
public:
  Step3 ();
  void run ();
  // Added Public methods
  void test_run(); // Here we run our tests!

  /*Johanna makes tests!:*/
  void test_outer_product(); //Test outer product with tensors!
  void test_MG(); //Test outer product with tensors!
  void presmooth_test(Vector<double> &dst, Vector<double> &src, const SparseMatrix<double> *&A);
  void GLT_init(); // Initialize the glt_precond. this is later put in solve();

private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void output_results () const;
  /*Added private methods */
  void inputFile(unsigned int size1, unsigned int size2,std::string filename);
  void inputFile_supplied(unsigned int, unsigned int,std::string,SparsityPattern &sp, SparseMatrix<double> &sm);
  void writeToFile(SparseMatrix<double> &matrix,std::string filename);
  /* Methods needed for initializing GLT */
  void kronProd(SparseMatrix<double> &A, SparseMatrix<double> &B,
    SparsityPattern &sp, SparseMatrix <double> &M);
  void kronProd_vector(Vector<double> &A, Vector<double> &B,
    SparsityPattern &sp, SparseMatrix <double> &M);
  void spdiags(double n,SparsityPattern &spaa, SparseMatrix <double> &aa);
  void transp(SparseMatrix<double> &A, SparsityPattern &sp, SparseMatrix<double> &M);
  void prol_const(double n, SparsityPattern &spP, SparseMatrix<double> &P);
  void transMultMult(SparseMatrix<double> &P, SparseMatrix<double> &b, SparsityPattern &sp, SparseMatrix<double> &sM);
  double vectorProd(Vector<int> v1, Vector<int> v2);
  Vector<int> factor(const int N);
  Vector<int> unique(Vector<int> factor);
  Vector<int> accumVector(Vector<int> v);
  const SparseMatrix<double> pointMatrix(SparsityPattern &sp);
  //const SparseMatrix<double> transMultMult(const SparseMatrix<double> &P,const SparseMatrix<double> *&b, int level, SparsityPattern &sp);
  //const SparseMatrix<double> prol_const(double n, SparsityPattern &spP);

  Triangulation<2>     triangulation;
  FE_Q<2>              fe;
  DoFHandler<2>        dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;
};
Step3::Step3 ()
:
fe (1),
dof_handler (triangulation)
{}
void Step3::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (5);
  std::cout << "Number of active cells: "
  << triangulation.n_active_cells()
  << std::endl;
}
void Step3::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  std::cout << "Number of degrees of freedom: "
  << dof_handler.n_dofs()
  << std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}
void Step3::assemble_system ()
{
  QGauss<2>  quadrature_formula(2);
  FEValues<2> fe_values (fe, quadrature_formula,
   update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();
  DoFHandler<2>::active_cell_iterator endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    cell_matrix = 0;
    cell_rhs = 0;
    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i){
        for (unsigned int j=0; j<dofs_per_cell; ++j){
          cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
           fe_values.shape_grad (j, q_index) *
           fe_values.JxW (q_index));
        }
      }
      for (unsigned int i=0; i<dofs_per_cell; ++i){
        cell_rhs(i) += (fe_values.shape_value (i, q_index) *
          1 *
          fe_values.JxW (q_index));
      }
    }
    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i){
      for (unsigned int j=0; j<dofs_per_cell; ++j){
        system_matrix.add (local_dof_indices[i],
         local_dof_indices[j],
         cell_matrix(i,j));
      }
    }
    for (unsigned int i=0; i<dofs_per_cell; ++i){
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }
  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
    0,
    ZeroFunction<2>(),
    boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
    system_matrix,
    solution,
    system_rhs);
}
void Step3::solve ()
{
      /* Original Step3::solve */ 
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
    PreconditionIdentity());
  
}
void Step3::output_results () const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output ("solution.gpl");
  data_out.write_gnuplot (output);
}
void Step3::run ()
{
      /* Original Step3::run */
  make_grid ();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
}

  /*======================================== Added Methods for tests ======================================================*/

       /* Read from input file ! !
        Adjust for writing on new sparsity pattern and Sparse*/
void Step3::inputFile(unsigned int size1, unsigned int size2,std::string filename){

  DynamicSparsityPattern dsp(size1,size2);
  std::ifstream input_file(filename);
  unsigned int i,j;
  double val;

  if(!input_file){
    std::cout << "While opening file, error encounterde"<<std::endl;
  }
  else{
    std::cout << "File is successfully opened" << std::endl;
  }
  while(input_file>>i>>j>>val)//(!input_file.eof())
  {
    dsp.add(i-1,j-1);
  }
  input_file.close();
  dsp.compress();
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  std::ifstream input_file_again(filename);
  if(!input_file_again){
    std::cout << "While opening a file an error is encountered" << std::endl;
  }
  else{
    std::cout << "File is successfully opened" << std::endl;
  }
  while(input_file_again>>i>>j>>val)//(!input_file.eof())
  {
    system_matrix.add(i-1,j-1,val);
  }
  input_file_again.close();

}

void Step3::inputFile_supplied(unsigned int size1, unsigned int size2,std::string filename, 
  SparsityPattern &sp, SparseMatrix<double> &sm){

  DynamicSparsityPattern dsp(size1,size2);
  std::ifstream input_file(filename);
  unsigned int i,j;
  double val;

  if(!input_file){
    std::cout << "While opening file, error encounterde"<<std::endl;
  }
  else{
    //std::cout << "File is successfully opened" << std::endl;
  }
  while(input_file>>i>>j>>val)//(!input_file.eof())
  {
    dsp.add(i-1,j-1);
  }
  input_file.close();
  dsp.compress();
  sp.copy_from(dsp);
  sm.reinit(sp);

  std::ifstream input_file_again(filename);
  if(!input_file_again){
    std::cout << "While opening a file an error is encountered" << std::endl;
  }
  else{
    //std::cout << "File is successfully opened" << std::endl;
  }
  while(input_file_again>>i>>j>>val)//(!input_file.eof())
  {
    sm.add(i-1,j-1,val);
  }
  input_file_again.close();
}

void Step3::writeToFile(SparseMatrix<double> &matrix,std::string filename){
  // filename ex "matrix.txt"
  std::ofstream out_system (filename);
  matrix.print_formatted(out_system,3,true,0," ",1); 
}

// This public method runs our tests !
void Step3::test_run(){
  SparsityPattern spAFin,spA;
  SparseMatrix<double> AFin,A;
  int size = 125;
  inputFile_supplied(size,size,"AFin.txt",spAFin,AFin);
  int N = size; // N could be an int, AFin quadratic.
  Vector<double> y;
  y.reinit(N);
  y = 1;
  Vector<double> b(N);
  AFin.vmult(b,y);  // Have to reinit b...
  //mgPrecondition mg(AFin,b);
  //mg.testPP();
}

/* This method is to test initializing the GLT.
  Its content will later be included in solve() 

  Note: We should remove system_matrix in mgPrecond and
  only work with BB, that way we halve the memory usage.

  */
void Step3::GLT_init(){

  int size = 125;
  int ref = 2;

  SparsityPattern *spP;
  SparsityPattern *spB;
  spP = new SparsityPattern [ref];
  spB = new SparsityPattern [ref+1];

  SparseMatrix<double> *BB;
  SparseMatrix<double> *PP;
  PP = new SparseMatrix<double> [ref];
  BB = new SparseMatrix<double> [ref+1]; // BB[0] = AFin
  inputFile_supplied(size,size,"AFin.txt",spB[0],BB[0]);

  Vector<double> y;
  y.reinit(size);
  y = 1;
  Vector<double> b(size);
  BB[0].vmult(b,y);  // Have to reinit b...

  int N = BB[0].m(); // = size
  Vector<int> NN = factor(N);
  Vector<int> nu = unique(NN);
  Vector<int> reps = accumVector(NN);
  double n = vectorProd(reps,nu);
  int level = 0;    
  //BB{i} = AFin;
  //PP{i} = 0;
  double n1 = n;

  while(n1>=3){
      // P = (1/n1)*prol([. . .][ . . .])
      //PP{i} = P
    prol_const(n1,spP[level],PP[level]);
    // B = P'*BB{i}*P
    // BB{i+1} = B
    transMultMult(PP[level],BB[level], spB[level+1], BB[level+1]);
    
    n1 = (n1+1)/2;
    level++;
  }
  /*
  std::cout<<" PP[0] mxn : "<<PP[0].m()<<" "<<PP[0].n()<<std::endl;
  std::cout<<" PP[1] mxn : "<<PP[1].m()<<" "<<PP[1].n()<<std::endl;
  std::cout<<" BB[0] mxn : "<<BB[0].m()<<" "<<BB[0].n()<<std::endl;
  std::cout<<" BB[1] mxn : "<<BB[1].m()<<" "<<BB[1].n()<<std::endl;
  std::cout<<" BB[2] mxn : "<<BB[2].m()<<" "<<BB[2].n()<<std::endl;
  */

  /*
  Solve . . .
  */

  // Freeing memory allocation  //Problem med att free:a BB när den används i mgPre!
  //delete[] PP;
  //delete[] BB;
  //delete[] spP;
  //delete[] spB
}

/*======================================== Johanna makes tests ======================================================*/
void Step3::test_outer_product(){
    //Vector v();

    //Tensor<>
}
void Step3::test_MG(){
  unsigned int n=3;

  DynamicSparsityPattern dspAFin(n,n);
  dspAFin.add(0,2);
  dspAFin.add(0,0);
  dspAFin.add(1,1);
  dspAFin.add(2,2);
  SparsityPattern spAFin;
  spAFin.copy_from(dspAFin);
  SparseMatrix<double> AFin(spAFin);
  AFin.add(0,2,4.0);
  AFin.add(0,0,1.0);
  AFin.add(1,1,3.0);
  AFin.add(2,2,1.0);

  Vector<double> b(n);
  b[0]=1;
  b[1]=2;
  b[2]=3;

  //mgPrecondition mg(AFin,b);

  //const
  //SparseMatrix<double> A;
  //A.copy_from(AFin);
  Vector<double> x(n);
  x=0;

  //residual test (ok)
/*  Vector<double> r(n);
  mg.newResidual(r,x,b,AFin); //residual  r=b-A*x

  std::cout<<" r0:   ";
  r.print(std::cout);
  std::cout<<"\n";

  x=1;

  mg.newResidual(r,x,b,AFin); //residual  r=b-A*x
  std::cout<<" r1:   ";
  r.print(std::cout);
  std::cout<<"\n";

  //result: residual works*/

  //presmooth test
/*  const SparseMatrix<double> *a(&AFin);
  std::vector<SparseMatrix<double> const *> BB;
  BB.push_back(a);

  mg.presmooth_test(x,b,a); //OBS: behöver matris utan nollelement på diagonalen
  std::cout<<"test av presmooth med direkt pointer ger resultat: "; //1.000e+00 6.667e-01 3.000e+00
  x.print(std::cout);
  std::cout<<"\n";

  x=0;
  mg.presmooth_test(x,b,BB[0]);
  std::cout<<"test av presmooth med pointer från std::vector ger resultat: "; //-1.100e+01 6.667e-01 3.000e+00 lolwut, måste vara att x inte resettats
  x.print(std::cout);
  std::cout<<"\n";*/

  std::vector<SparseMatrix<double> const *> BB;
  BB.push_back(&AFin);
  presmooth_test(x,b,BB[0]);
}

void Step3::presmooth_test(Vector<double> &dst,Vector<double> &src,const SparseMatrix<double> *&A){
//    void Step3::presmooth_test(Vector<double> &dst,Vector<double> &src,const SparseMatrix<double>* &A){
  A->Jacobi_step(dst,src,1);
}

/*========================================== Methods added for GLT_init ==============================================================================*/

/* Vector A is percieved to be a 1xn vector
and Vector B is percieved to be a mx1 vector!
        resulting Matrix M is a mxn matrix
*/
void Step3::kronProd_vector(Vector<double> &A, Vector<double> &B,
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

/*
        This is the kronecker product of 2 sparse matrices
*/
void Step3::kronProd(SparseMatrix<double> &A, SparseMatrix<double> &B,
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
  SparseMatrix<double>::iterator itB = B.begin(); // I fixed it ;P
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
/* Here we implement the prol([2 1 1],[0 -1 1],n)
        dd = [0 -1 1]
        PP = [2 1 1]
        performs the operation:
        aa = spdiags(kron(PP,one(n,1),dd,n,n));
*/
void Step3::spdiags(double n,SparsityPattern &spaa, SparseMatrix <double> &aa){
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
    aa.add(i,i,sm1(i,0));   // Column 0 on diag 0
    aa.add(i+1,i,sm1(i+1,1)); // Column 1 on diag -1
    aa.add(i,i+1,sm1(i,2));   // Column 2 on diag 1
  }
  aa.add(N-1,N-1,sm1(N-1,0));
}

/*  Here we make a transpose of matrix A which can be any type of a matrix
        output sp and M */
void Step3::transp(SparseMatrix<double> &A, SparsityPattern &sp, SparseMatrix<double> &M){
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

/* This method performs the prod(nu.^(resp(nu)./3))
        action given in the GLTmg*/
double Step3::vectorProd(Vector<int> reps, Vector<int> nu){
  double n = 1;
    // the reps(nu) action
  Vector<double> repsNu;  //Used for development
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

/* Return a vector containing the primenumbers of N
        Might Implement Sieve at a later point    */
Vector<int> Step3::factor(int N){
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
Vector<int> Step3::unique(Vector<int> factor){
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
Vector<int> Step3::accumVector(Vector<int> v){
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

/* This Method performs the B = P'*BB{i}*P operation, where BB is a std::vector
    and a member of mgPrecondtion. The method returns the const matrix P */
void Step3::transMultMult(SparseMatrix<double> &P, SparseMatrix<double> &b, SparsityPattern &sp,
  SparseMatrix<double> &sm){
  SparsityPattern spTemp;
  SparseMatrix<double> temp;
  
  DynamicSparsityPattern dspTemp(0),dspB(0);
  spTemp.copy_from(dspTemp);
  temp.reinit(spTemp);

  b.mmult(temp, P, Vector<double>(), true);

  sp.copy_from(dspB);
  sm.reinit(sp);
  P.Tmmult(sm,temp, Vector<double>(),true);
}

/* This is an implementation of the algorithm
        (1/n)*prol([2 1 1], [0 -1 1],n)
        given in the gltmg_test matlab code

        This code is memory-ineffective and should include destructors

        Method tested and seems to work.... Finaly....

  We change the prol method such that it can return a const SparseMatrix :)
 */
void Step3::prol_const(double n, SparsityPattern &spP,SparseMatrix<double> &P){
  // aa = spdiags(kron(PP,ones),dd,n,n)
  SparsityPattern spaa;
  SparseMatrix<double> aa;
  spdiags(n,spaa,aa);
  // smP = kron(aa,kron(aa,aa))
  SparsityPattern spTemp;
  SparseMatrix<double> smTemp;
  kronProd(aa,aa,spTemp,smTemp);
  SparsityPattern spP2;
  SparseMatrix<double> smP;
  kronProd(aa,smTemp,spP2,smP);
  // Create smH = speye(n) then remove every other row
  // smH = smH(1:2:n,:)
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
  // H = kron(smH,kron(smH,smH))
  SparsityPattern sp_dummyH;
  SparseMatrix<double> dummyH; 
  SparsityPattern spH;
  SparseMatrix<double> H;
  kronProd(smH,smH,sp_dummyH,dummyH);
  kronProd(smH,dummyH,spH,H);
  // P = smP*H'; P = (1/n)*P
  SparsityPattern spTranspH;
  SparseMatrix<double> transpH;
  //SparseMatrix<double> P;
  transp(H,spTranspH,transpH);
  DynamicSparsityPattern dspP(0);
  spP.copy_from(dspP);
  P.reinit(spP);
  smP.mmult(P,transpH,Vector<double>(),true);
  //P = (1/n).P
  P*=(1/n);
}

/* Dummy version 
const SparseMatrix<double> Step3::prol_const(double n, SparsityPattern &spP){
  // aa = spdiags(kron(PP,ones),dd,n,n)
  SparsityPattern spaa;
  SparseMatrix<double> aa;
  spdiags(n,spaa,aa);
  // smP = kron(aa,kron(aa,aa))
  SparsityPattern spTemp;
  SparseMatrix<double> smTemp;
  kronProd(aa,aa,spTemp,smTemp);
  SparsityPattern spP2;
  SparseMatrix<double> smP;
  kronProd(aa,smTemp,spP2,smP);
  // Create smH = speye(n) then remove every other row
  //  smH = smH(1:2:n,:)
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
  // H = kron(smH,kron(smH,smH))
  SparsityPattern sp_dummyH;
  SparseMatrix<double> dummyH; 
  SparsityPattern spH;
  SparseMatrix<double> H;
  kronProd(smH,smH,sp_dummyH,dummyH);
  kronProd(smH,dummyH,spH,H);
  // P = smP*H'; P = (1/n)*P
  SparsityPattern spTranspH;
  SparseMatrix<double> transpH;
  SparseMatrix<double> P;
  transp(H,spTranspH,transpH);
  DynamicSparsityPattern dspP(0);
  spP.copy_from(dspP);
  P.reinit(spP);
  smP.mmult(P,transpH,Vector<double>(),true);
  //P = (1/n).P
  P*=(1/n);

  return P;
}

*/
/* ============================================ These methods have been ommited ============================================ */

/* Method that returns a double matrix with only one zero element */
const SparseMatrix<double> Step3::pointMatrix(SparsityPattern &sp){
  SparseMatrix<double> M;
  DynamicSparsityPattern dsp(1,1);
  dsp.add(0,0);
  dsp.compress();
  sp.copy_from(dsp);
  M.reinit(sp);
  M.add(0,0,0);
  return M;
}
/*=====================================================================================================================================================*/

int main ()
{
  //deallog.depth_console (2);
  //Step3 laplace_problem;
  //laplace_problem.run ();
  Step3 test;
  //test.run();
  test.GLT_init();
  //test.test_MG();

  return 0;
}
/* - - - - - -  The code-dump - - - - - - - - - */


  /* Problem with dim, but this seems to work . . .
  std::vector<SparseMatrix<double> const *> BB;
  std::vector<SparseMatrix<double> const *>::iterator it;
  double n1 = 5;
  SparsityPattern spB;
  SparsityPattern spB2;
  const SparseMatrix<double> B = mg.prol_const(n1,spB);
  const SparseMatrix<double> B2 = mg.prol_const(n1,spB);
  BB.push_back(&B);
  BB.push_back(&B2);
  SparsityPattern spfoo;
  SparseMatrix<double> foo;
  DynamicSparsityPattern dsp(0);
  spfoo.copy_from(dsp);
  foo.reinit(spfoo);
  BB[0]->mmult(foo,*BB[1],Vector<double>(),true);
 */ 

/* This is a test of the prol method! in mgPrecondtion.h
  inputFile_supplied(3,3,"A.txt",spA,A);
  //A.print_formatted(std::cout,1,true,0," ",1);
  SparsityPattern spFoo;
  SparseMatrix<double> foo;
  std::cout<<" ======================== "<<std::endl;
  double n = 5;
  mg.prol(n,spFoo,foo);
*/

/* This is how you play with tensors*/
/*
  std::cout<<" - - - - - - - - - - - - - - - "<<std::endl;
  double foo[4] = {0, 1, 2, 3};
  Tensor<2,2,double> ten();
  int 
  std::cout<<" ten: "<<ten.memory_consumption()<<std::endl;
*/

/* This is how to initialize a matrix */
/*
  DynamicSparsityPattern dspM(0);  // Must initialize before using
  //dspM.add(1,1);
  spM.copy_from(dspM);
  M.reinit(spM);
  AFin.mmult(M,A,Vector<double>(),true);
  M.print_forma tted(std::cout,1,true,0," ",1);
*/
/* - - - - - Here I test BB and PP storages - - - - - - -*/
  /* Not sure if this is sound memory handling! ! ! 
      But this works just fine . . . . 
  std::vector<SparseMatrix<double>*> vicky;
  vicky.push_back(&A); vicky.push_back(&B);
  vicky[0]->print_formatted(std::cout,1,true,0," ",1);
      //const SparseMatrix<double> *m = system_matrix;
    //BB.push_back(m);
  */

  /* Could use MGLevelObject<Object>, which actually referenses the std::vector!
    Cant find the method to add shit :/

  MGLevelObject<SparseMatrix <double>> list();
  list.add(A);*/


  /* Part of the GLTmg Run 
  Vector<int> NN = mgTest.factor(N);
  Vector<int> nu = mgTest.unique(NN);
  Vector<int> reps = mgTest.accumVector(NN);
  std::cout<<" NN:   "; NN.print(std::cout,3,true,true);
  std::cout<<" nu:   "; nu.print(std::cout,3,true,true);
  std::cout<<" reps: "; reps.print(std::cout,3,true,true);
  double n = mgTest.vectorProd(reps,nu);
  std::cout<<"n : "<<n<<std::endl;
  */

/*
   SM.print_formatted(std::cout,1,true,0," ",1);
  writeToFile(SM,"matrix.txt");
 SparsityPattern::iterator start = SP.begin();

  system_matrix.print_formatted(std::cout,3,true,0," ",1);
  system_rhs.print(std::cout,3,true,true);
  inputFile(3,3,"bb.txt");
  mgPrecondition mgPre1(system_matrix,system_rhs);
  mgPre1.printMatrix();
  system_matrix.print_formatted(std::cout,1,true,0," ",1);
  mgPre1.printVector();

  Vector<double> test_vector;
  canITouchThis(test_vector);
  //test_vector.print(std::cout,3,true,true);
  Vector<double>::iterator iter = b.begin();
  Vector<double>::iterator ender = b.end();
  for(; iter!=ender; ++iter){
    std::cout<<*iter<<std::endl;
    *iter = 10;
  }
  b.print(std::cout,1,true,true);
  */
