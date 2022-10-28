#include "stdio.h"
#include "math.h"
#include "kfft.cpp"

// 200Hz 512N 9Hz
// 200Hz 256N 18Hz

#define PI 3.1415926535
#define Fs 200
#define N 512

int main()
{ 
	int i,j;
    double pr[N],pi[N],fr[N],fi[N],t[N];
    for (i=0; i<N; i++)  //生成输入信号
    { 
		// t[i] = i*0.001;
        t[i] = i*1.0/Fs;
		// pr[i]=1.2+2.7*cos(2*PI*33*t[i])+5*cos(2*PI*200*t[i]+PI/2);
        pr[i]=10+100*cos(2*PI*9*t[i]);
        pi[i]=0.0;
	}
		
    kfft(pr,pi,N,9,fr,fi);  //调用FFT函数
	for (i=0; i<100; i++)
    { 
        printf("%d\t%lf\n",i,pr[i]); //输出结果
    }
    // system("pause");
    return 0;
}