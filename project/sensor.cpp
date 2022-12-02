#include "sensor.h"
using namespace std;


sensor::sensor()
{
}

sensor::~sensor()
{
}

// JYRM3100数据解析
void sensor::JYRM3100_GetH(string &input, vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata) {
    // 数据标签
    vector<string> Hlabels = {"Magx:", "Magy:", "Magz:"};
    // 存储一组数据
    vector<double> Hdatas(3);
    int len = input.length();
    // 初始化数据指针
    int n = input.find(Hlabels[0]);
    for(int j = 0; j < Hxdata.size(); j++) {
        for(int i = 0; i < Hlabels.size(); i++) {
            double data = 0;
            double sign = 1;
            // 找到标签位置
            n = input.find(Hlabels[i], n);
            // 指针向后移动到数据位
            n += Hlabels[i].length();
            while(input[n] != ',' && n < len) {
                // 处理负数
                if(input[n] == '-') {
                    sign = -1;
                }
                else {
                    data = data * 10.0 + (input[n] - '0');
                }
                n++;
            }
            Hdatas[i] = sign * data;
        }
        // 储存一组数据
        Hxdata[j] = Hdatas[0];
        Hydata[j] = Hdatas[1];
        Hzdata[j] = Hdatas[2];
    }
}

// JY901获取姿态角
void sensor::JY901_GetAngle(string &input, double &x_roll, double &y_pitch, double &z_yaw) {
    int n = input.length();
    int start = 0;
    while(start <= n - 2) {
        // 寻找起始位置
        if(input[start] == 0x55 && input[start + 1] == 0x53) {
            break;
        }
        start++;
    }

    x_roll   = (input[start + 3] << 8 | (unsigned char)input[start + 2]) / 32768.0 * 180;    // Roll=((RollH<<8)|RollL)/32768*180(°)
    y_pitch  = (input[start + 5] << 8 | (unsigned char)input[start + 4]) / 32768.0 * 180;    // Pitch=((PitchH<<8)|PitchL)/32768*180(°)
    z_yaw    = (input[start + 7] << 8 | (unsigned char)input[start + 6]) / 32768.0 * 180;    // Yaw=((YawH<<8)|YawL)/32768*180(°)
}

// JY901磁场数据解析
void sensor::JY901_GetH(string &input, vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata) {
    int n = input.length();
    int start = 0;
    while(start <= n - 2) {
        // 寻找起始位置
        if(input[start] == 0x55 && input[start + 1] == 0x54) {
            break;
        }
        start++;
    }

    // 储存磁场数据
    for(int i = 0; i < Hxdata.size(); i++) {
        if(start > n - 11) break;
        // 检查合法性
        if(input[start] != 0x55 || input[start + 1] != 0x54) {
            break;
        }
        double Hx = input[start + 3] << 8 | (unsigned char)input[start + 2];    // Hx=((HxH<<8)|HxL)
        double Hy = input[start + 5] << 8 | (unsigned char)input[start + 4];    // Hy=((HyH<<8)|HyL)
        double Hz = input[start + 7] << 8 | (unsigned char)input[start + 6];    // Hz=((HzH<<8)|HzL)
        Hxdata[i] = Hx;
        Hydata[i] = Hy;
        Hzdata[i] = Hz;   // 记录一组数据
        start += 22;    // 指针移到下一组数据
    }
}

