#ifndef _SENSOR_H_
#define _SENSOR_H_

#include <iostream>
#include <vector>
using namespace std;

class sensor
{
private:
    
public:
    sensor();
    ~sensor();
    // JYRM3100数据解析
    void JYRM3100_GetH(string &input, vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata);

    // JY901获取姿态角
    void JY901_GetAngle(string &input, double &x_roll, double &y_pitch, double &z_yaw);
    
    // JY901磁场数据解析
    void JY901_GetH(string &input, vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata);
};


#endif
