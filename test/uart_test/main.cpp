#include "WZSerialPort.h"
#include "../filter_test/IIR.h"
#include "../filter_test/filter.cpp"
#include "../fft_test/kfft.cpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#include <math.h>
#include <windows.h>
using namespace std;

#define PhysicalTest    // 实物测试
#define Simulation      // 仿真测试

#define PI 3.1415926535
void Simulate(int Fs, int N, vector<double> &Hdata, vector<double> &Hidata) {
    default_random_engine r;
    normal_distribution<double> u(0,1); // 均值为0，标准差为1的随机数
    r.seed(time(0));
    for (int i = 0; i < N; i++)  //生成输入信号
    {
        Hdata.push_back(10 + 100*cos(2*PI*1.95*(i*1.0/Fs)) + u(r));
        Hidata.push_back(0.0);
	}
}

// 字符串转换16进制
string binaryToHex(const string& binaryStr)
{
    string ret;
    static const char *hex = "0123456789ABCDEF";
    for (auto c:binaryStr)
    {
        ret.push_back(hex[(c >> 4) & 0xf]); //取二进制高四位
        ret.push_back(hex[c & 0xf]);        //取二进制低四位
    }
    return ret;
}

// 数据转换函数
void DataTransfer(string &input, vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata) {
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
    while(start <= n - 11) {
        // 检查合法性
        if(input[start] != 0x55 || input[start + 1] != 0x54) {
            break;
        }
        double Hx = input[start + 3] << 8 | (unsigned char)input[start + 2];    // Hx=((HxH<<8)|HxL)
        double Hy = input[start + 5] << 8 | (unsigned char)input[start + 4];    // Hy=((HyH<<8)|HyL)
        double Hz = input[start + 7] << 8 | (unsigned char)input[start + 6];    // Hz=((HzH<<8)|HzL)
        Hxdata.push_back(Hx);
        Hydata.push_back(Hy);
        Hzdata.push_back(Hz);   // 记录一组数据 
        start += 22;    // 指针移到下一组数据
    }
}

// 数据存储函数
void DataStorage(vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata, ofstream &fout) {
    if(!Hxdata.size() | !Hydata.size() |!Hzdata.size()) {
        return;
    }
    int n = Hxdata.size();
    for(int i = 0; i < n; i++) {
        fout << Hxdata[i] << ",";
        fout << Hydata[i] << ",";
        fout << Hzdata[i] << endl;
    }
}

// 获取姿态角
void GetAngle(string &input, double &x_roll, double &y_pitch, double &z_yaw) {
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


void getH(string &input, vector<double> &Hxdata, vector<double> &Hydata, vector<double> &Hzdata) {
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

int main()
{
    string OriginalData;

    vector<double> Hxdata(512);
    vector<double> Hydata(512);
    vector<double> Hzdata(512);
    double x_roll;
    double y_pitch;
    double z_yaw;

    ofstream fout;
    WZSerialPort w;
    // IIR_Filter IIR;

// 实物测试部分
#ifdef PhysicalTest
#undef Simulation
    vector<double> Hidata(512, 0);
    vector<double> fr(512), fi(512);
	
	if (w.open("COM7"))
	{
        string cmd_unlock = {(char)0xff, (char)0xaa, 0x69, (char)0x88, (char)0xb5};
		string cmd_endonce = {(char)0xff, (char)0xaa, 0x03, (char)0x0c, (char)0x00};
        // cout << "SEND UNLOCK!" << endl;
        // w.send(cmd_unlock);
        // Sleep(50);
        // cout << "SEND GETONCE!" << endl;
		// w.send(cmd_endonce);
        // 取数据
        OriginalData = w.receive();
        // cout << OriginalData << endl;
        getH(OriginalData, Hxdata, Hydata, Hzdata);

        // GetAngle(OriginalData, x_roll, y_pitch, z_yaw);
        // cout << x_roll << ", " << y_pitch << ", " << z_yaw << endl;

        // // 转换数据并存储
        // DataTransfer(OriginalData, Hxdata, Hydata, Hzdata);
        fout.open("C:/code/code/Magnetic-beacon-positioning/test/uart_test/Hdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        // // 对磁场数据滤波并保存
        // IIR.Filter(Hxdata);
        // IIR.Filter(Hydata);
        // IIR.Filter(Hzdata);
        // // LowPassFilter(Hxdata, 5, 200);
        // // LowPassFilter(Hydata, 5, 200);
        // // LowPassFilter(Hzdata, 5, 200);
        // fout.open("C:/code/code/Magnetic-beacon-positioning/test/uart_test/Filterdata.txt");
        // DataStorage(Hxdata, Hydata, Hzdata, fout);
        // fout.close();

        // // 对磁场傅里叶变换并存储
        // kfft(Hxdata, Hidata, 512, 9, fr, fi);
        // kfft(Hydata, Hidata, 512, 9, fr, fi);
        // kfft(Hzdata, Hidata, 512, 9, fr, fi);
        // fout.open("C:/code/code/Magnetic-beacon-positioning/test/uart_test/FFTdata.txt");
        // DataStorage(Hxdata, Hydata, Hzdata, fout);
        // fout.close();

        // // 以16进制存储原始数据
        // // fout << binaryToHex(OriginalData).c_str() << endl;
        // fout.close();
		w.close();
	}
    cout << "save successfully!" << endl;
#endif


// 仿真模拟部分
#ifdef Simulation
    int fs = 200;
    int n = 512;
    vector<double> Hdata, Hidata;
    vector<double> fr(n), fi(n);
    Simulate(fs, n, Hdata, Hidata);
    kfft(Hdata, Hidata, n, 9, fr, fi);
    for (int i=0; i<10; i++)
    { 
        printf("%d\t%lf\n",i,Hdata[i]); //输出结果
    }
#endif
    // system("pause");
	return 0;
}