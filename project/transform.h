#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.h"
using namespace std;

#ifndef PI
#define PI 3.1415926535
#endif

void getN(vector<vector<double>>& N, double x, double y, double z);
void tranform(vector<vector<double>>& E, double x, double y, double z);

#endif
