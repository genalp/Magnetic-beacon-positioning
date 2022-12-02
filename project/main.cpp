#include "WZSerialPort.h"
#include "IIR.h"
#include "kfft.h"
#include "matrix.h"
#include "transform.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#include <math.h>
using namespace std;

#define PhysicalTest    // 实物测试
#define Simulation      // 仿真测试

#define PI 3.1415926535
#define Point_num 512

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

// 实物测试部分
#ifdef PhysicalTest
#undef Simulation
    vector<double> Hidata(Point_num, 0);
    vector<double> fr(Point_num), fi(Point_num);
	
	if (w.open("COM7"))
	{
		// string str = "hello";
		// w.send(str);
        // 取数据
        OriginalData = w.receive();

        // 获取角度
        // GetAngle(OriginalData, x_roll, y_pitch, z_yaw);
        // cout << x_roll << ", " << y_pitch << ", " << z_yaw << endl;

        // 转换数据并存储
        // DataTransfer(OriginalData, Hxdata, Hydata, Hzdata);
        getH(OriginalData, Hxdata, Hydata, Hzdata);
        fout.open("./Hdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        // 对磁场数据滤波并保存
        IIR.Filter(Hxdata);
        IIR.Filter(Hydata);
        IIR.Filter(Hzdata);
        fout.open("./Filterdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        // 对磁场傅里叶变换并存储
        int ftt_n = (int)log2(Point_num);
        kfft(Hxdata, Hidata, Point_num, ftt_n, fr, fi);
        kfft(Hydata, Hidata, Point_num, ftt_n, fr, fi);
        kfft(Hzdata, Hidata, Point_num, ftt_n, fr, fi);
        fout.open("./FFTdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        cout << Hxdata[23] << ", " << Hydata[23] << ", " << Hzdata[23] << endl;

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