#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <math.h>
#include <vector>
using namespace std;
vector<vector<double>> creatmatrix(int h,int l);
vector<vector<double>> matrix_plus(const vector<vector<double>>&A,const vector<vector<double>>&B);
vector<vector<double>> matrix_minus(const vector<vector<double>>&A,const vector<vector<double>>&B);
vector<vector<double>> matrix_multiply(const vector<vector<double>>&A,const vector<vector<double>>&B);
vector<vector<double>> matrix_multiply_num(const vector<vector<double>>&A,double num);
vector<vector<double>> matrix_overlaying_below(const vector<vector<double>>&A,const vector<vector<double>>&B);
vector<vector<double>> matrix_overlaying_beside(const vector<vector<double>>&A,const vector<vector<double>>&B);
vector<vector<double>> matrix_trans(const vector<vector<double>> &A);
vector<vector<double>> matrix_inverse(const vector<vector<double>> &A);
void show_matrix(const vector<vector<double>> &A);

#endif