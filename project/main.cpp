#include "WZSerialPort.h"
#include "IIR.h"
#include "kfft.h"
#include "matrix.h"
#include "transform.h"
#include "sensor.h"
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
#define Point_num 512   // 采样点数

// 传感器操作命令
#define JYRM3100_StartSend  "AT+PRATE=10\r\n"   // JYRM3100 开始以100Hz发送
#define JYRM3100_StopSend   "AT+PRATE=0\r\n"    // JYRM3100 停止发送（单次发送）
#define JY901_Unlock        {(char)0xff, (char)0xaa, (char)0x69, (char)0x88, (char)0xb5}    // JY901 串口操作解锁
#define JY901_Sendonce      {(char)0xff, (char)0xaa, (char)0x03, (char)0x0c, (char)0x00}    // JY901 单次发送数据

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


int main()
{
    string OriginalData;

    vector<double> Hxdata(Point_num);
    vector<double> Hydata(Point_num);
    vector<double> Hzdata(Point_num);
    double x_roll;
    double y_pitch;
    double z_yaw;
    Matrix HM;
    vector<vector<double>> E = HM.creatmatrix(3,1);

    ofstream fout;
    WZSerialPort w;
    IIR_Filter IIR;
    sensor JYSensor;


// 实物测试部分
#ifdef PhysicalTest
#undef Simulation
    vector<double> Hidata(Point_num, 0);
    vector<double> fr(Point_num), fi(Point_num);
	
	if (w.open("COM6"))
	{
        // 发送取值命令
        // JYRM3100命令
        // w.send(JYRM3100_StartSend);
        // JY901命令
        w.send(JY901_Unlock);
        Sleep(50);
        w.send(JY901_Sendonce);
        
        // 取数据
        // OriginalData = w.receive(23085);
        OriginalData = w.receive(22);

        // 获取角度
        // GetAngle(OriginalData, x_roll, y_pitch, z_yaw);
        JYSensor.JY901_GetAngle(OriginalData, x_roll, y_pitch, z_yaw);
        cout << x_roll << ", " << y_pitch << ", " << z_yaw << endl;

        // // 转换数据并存储
        // JYSensor.JYRM3100_GetH(OriginalData, Hxdata, Hydata, Hzdata);
        // fout.open("./Hdata.txt");
        // DataStorage(Hxdata, Hydata, Hzdata, fout);
        // fout.close();

        // // 对磁场数据滤波并保存
        // IIR.Filter(Hxdata);
        // IIR.Filter(Hydata);
        // IIR.Filter(Hzdata);
        // fout.open("./Filterdata.txt");
        // DataStorage(Hxdata, Hydata, Hzdata, fout);
        // fout.close();

        // // 对磁场傅里叶变换并存储
        // int ftt_n = (int)log2(Point_num);
        // kfft(Hxdata, Hidata, Point_num, ftt_n, fr, fi);
        // kfft(Hydata, Hidata, Point_num, ftt_n, fr, fi);
        // kfft(Hzdata, Hidata, Point_num, ftt_n, fr, fi);
        // fout.open("./FFTdata.txt");
        // DataStorage(Hxdata, Hydata, Hzdata, fout);
        // fout.close();

        // cout << Hxdata[23] << ", " << Hydata[23] << ", " << Hzdata[23] << endl;

        // // 坐标转换测试
        // E[0][0] = Hxdata[5]/256;
        // E[1][0] = Hydata[5]/256;
        // E[2][0] = Hzdata[5]/256;
        // tranform(E, x_roll, y_pitch, z_yaw);
        // // HM.show_matrix(E);
        // cout << E[0][0]*E[0][0] + E[1][0]*E[1][0] + E[2][0]*E[2][0] << endl;

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