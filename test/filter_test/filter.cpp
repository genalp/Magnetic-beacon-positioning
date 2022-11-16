#include <iostream>
#include <math.h>
#include <vector>
using namespace std;

#ifndef PI
#define PI 3.1415926535
#endif
#define FL 3    // 截止频率

float geta(int fl, int fs) {
    float t = 1.0/fs; // 采样间隔
    float a = fl * (2 * PI * t);    // fL = a / (2 * PI * t)
    return a;
}

void LowPassFilter(vector<double> &datas, int fl, int fs) {
    if(datas.size() < 2) return;
    float a = geta(fl, fs);
    for(int i = 1; i < datas.size(); i++) {
        datas[i] = a * datas[i] + (1 - a) * datas[i - 1];
    }
}