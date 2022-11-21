#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <math.h>
#include <vector>
using namespace std;

class Matrix
{
private:
    const double epsilon=1e-12;  //小于该数判断为0
public:
    Matrix();
    ~Matrix();
    //创建 h行l列的矩阵，并将初始各值设定为0
    vector<vector<double>> creatmatrix(int h,int l);

    //矩阵A+矩阵B=矩阵C，并返回
    vector<vector<double>> matrix_plus(const vector<vector<double>>&A,const vector<vector<double>>&B);

    //矩阵A-矩阵B=矩阵C，并返回
    vector<vector<double>> matrix_minus(const vector<vector<double>>&A,const vector<vector<double>>&B);

    //矩阵A*矩阵B=矩阵C，并返回
    vector<vector<double>> matrix_multiply(const vector<vector<double>>&A,const vector<vector<double>>&B);

    //矩阵A*num=矩阵B，并返回
    vector<vector<double>> matrix_multiply_num(const vector<vector<double>>&A,double num);

    //矩阵A与矩阵B上下叠加获得新的矩阵C,并返回
    vector<vector<double>> matrix_overlaying_below(const vector<vector<double>>&A,const vector<vector<double>>&B);

    //矩阵A与矩阵B左右叠加，获得新的矩阵C，并返回
    vector<vector<double>> matrix_overlaying_beside(const vector<vector<double>>&A,const vector<vector<double>>&B);

    //输入矩阵A，输出矩阵A的转置矩阵AT
    vector<vector<double>> matrix_trans(const vector<vector<double>> &A);

    //输入矩阵A,输出矩阵A的逆矩阵inv_A
    vector<vector<double>> matrix_inverse(const vector<vector<double>> &A);
    void show_matrix(const vector<vector<double>> &A);
};

#endif