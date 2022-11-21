#include "WZSerialPort.h"
#include "IIR.h"
// #include "kfft.cpp"
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


void kfft(vector<double> &pr, vector<double> &pi, int n, int k, vector<double> &fr, vector<double> &fi)
{ 
    int it,m,is,i,j,nv,l0;
    double p,q,s,vr,vi,poddr,poddi;
    for (it=0; it<=n-1; it++)  //将pr[0]和pi[0]循环赋值给fr[]和fi[]
    { 
        m=it; 
        is=0;
        for(i=0; i<=k-1; i++)
        { 
            j=m/2; 
            is=2*is+(m-2*j); 
            m=j;
        }
        fr[it]=pr[is]; 
        fi[it]=pi[is];
    }
    pr[0]=1.0; 
    pi[0]=0.0;
    p=6.283185306/(1.0*n);
    pr[1]=cos(p); //将w=e^-j2pi/n用欧拉公式表示
    pi[1]=-sin(p);

    for (i=2; i<=n-1; i++)  //计算pr[]
    { 
        p=pr[i-1]*pr[1]; 
        q=pi[i-1]*pi[1];
        s=(pr[i-1]+pi[i-1])*(pr[1]+pi[1]);
        pr[i]=p-q; pi[i]=s-p-q;
    }
    for (it=0; it<=n-2; it=it+2)  
    { 
        vr=fr[it]; 
        vi=fi[it];
        fr[it]=vr+fr[it+1]; 
        fi[it]=vi+fi[it+1];
        fr[it+1]=vr-fr[it+1]; 
        fi[it+1]=vi-fi[it+1];
    }
    m=n/2; 
    nv=2;
    for (l0=k-2; l0>=0; l0--) //蝴蝶操作
    { 
        m=m/2; 
        nv=2*nv;
        for (it=0; it<=(m-1)*nv; it=it+nv)
        for (j=0; j<=(nv/2)-1; j++)
            { 
                p=pr[m*j]*fr[it+j+nv/2];
                q=pi[m*j]*fi[it+j+nv/2];
                s=pr[m*j]+pi[m*j];
                s=s*(fr[it+j+nv/2]+fi[it+j+nv/2]);
                poddr=p-q; 
                poddi=s-p-q;
                fr[it+j+nv/2]=fr[it+j]-poddr;
                fi[it+j+nv/2]=fi[it+j]-poddi;
                fr[it+j]=fr[it+j]+poddr;
                fi[it+j]=fi[it+j]+poddi;
            }
    }
    for (i=0; i<=n-1; i++)
    { 
        pr[i]=sqrt(fr[i]*fr[i]+fi[i]*fi[i]);  //幅值计算
    }
    return;
}



int main()
{
    string OriginalData;

    vector<double> Hxdata;
    vector<double> Hydata;
    vector<double> Hzdata;

    ofstream fout;
    WZSerialPort w;
    IIR_Filter IIR;

// 实物测试部分
#ifdef PhysicalTest
#undef Simulation
    vector<double> Hidata(512, 0);
    vector<double> fr(512), fi(512);
	
	if (w.open("COM6"))
	{
		// string str = "hello";
		// w.send(str);
        // 取数据
        OriginalData = w.receive();

        // 转换数据并存储
        DataTransfer(OriginalData, Hxdata, Hydata, Hzdata);
        fout.open("./Hdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        // 对磁场数据滤波并保存
        IIR.Filter(Hxdata);
        IIR.Filter(Hydata);
        IIR.Filter(Hzdata);
        // LowPassFilter(Hxdata, 5, 200);
        // LowPassFilter(Hydata, 5, 200);
        // LowPassFilter(Hzdata, 5, 200);
        fout.open("./Filterdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        // 对磁场傅里叶变换并存储
        kfft(Hxdata, Hidata, 512, 9, fr, fi);
        kfft(Hydata, Hidata, 512, 9, fr, fi);
        kfft(Hzdata, Hidata, 512, 9, fr, fi);
        fout.open("./FFTdata.txt");
        DataStorage(Hxdata, Hydata, Hzdata, fout);
        fout.close();

        // 以16进制存储原始数据
        // fout << binaryToHex(OriginalData).c_str() << endl;
        fout.close();
		w.close();
	}
    cout << "save successfully!" << endl;
    // system("C:\\code\\code\\Magnetic-beacon-positioning\\test\\uart_test\\Hdata.txt");
    // system("C:\\code\\code\\Magnetic-beacon-positioning\\test\\uart_test\\FFTdata.txt");
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