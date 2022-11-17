#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.h"
using namespace std;

void getN(vector<vector<double>>& N, double x, double y, double z) {
    Matrix M;
    vector<vector<double>> Rx = M.creatmatrix(3,3);
    vector<vector<double>> Ry = M.creatmatrix(3,3);
    vector<vector<double>> Rz = M.creatmatrix(3,3);
    
    // 构建Rx
    Rx[0][0] = 1;
    Rx[1][1] = cos(x);
    Rx[1][2] = sin(x);
    Rx[2][1] = -sin(x);
    Rx[2][2] = cos(x);

    // 构建Ry
    Ry[0][0] = cos(y);
    Ry[0][2] = -sin(y);
    Ry[1][1] = 1;
    Ry[2][0] = sin(y);
    Ry[2][2] = cos(y);

    // 构建Rz
    Rz[0][0] = cos(z);
    Rz[0][1] = sin(z);
    Rz[1][0] = -sin(z);
    Rz[1][1] = cos(z);
    Rz[2][2] = 1;

    N = M.matrix_multiply(Rx,Ry);
    N = M.matrix_multiply(N, Rz);
}


#define PI 3.1415926535

int main() {
    double x = 80.793 *PI/180;
    double y = -56.058 *PI/180;
    double z = 127.211 *PI/180;
    Matrix M;
    vector<vector<double>> N = M.creatmatrix(3,3);
    vector<vector<double>> E = M.creatmatrix(3,1);
    E[0][0] = -71;
    E[1][0] = -221;
    E[2][0] = 67;
    
    getN(N, x, y, z);
    N = M.matrix_trans(N);
    E = M.matrix_multiply(N, E);
    M.show_matrix(E);

    return 0;
}
