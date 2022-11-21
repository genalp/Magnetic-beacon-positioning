#ifndef _IIR_H_
#define _IIR_H_

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <vector>
#include <iostream>
using namespace std;

#ifndef PI
#define PI 3.1415926535
#endif
 
typedef struct 
{
    double Real_part;
    double Imag_Part;
} COMPLEX;
 
struct DESIGN_SPECIFICATION
{
    double Cotoff;   
    double Stopband;
    double Stopband_attenuation;
};

class IIR_Filter
{
public:
    IIR_Filter();
    ~IIR_Filter();

    int Ceil(double input);
    int Complex_Multiple(COMPLEX a,COMPLEX b,double *Res_Real,double *Res_Imag);
    int Complex_Division(COMPLEX a,COMPLEX b,double *Res_Real,double *Res_Imag);
    double Complex_Abs(COMPLEX a);
    double IIRFilter  (double *a, int Lenth_a,
                           double *b, int Lenth_b,
                           double Input_Data,
                           double *Memory_Buffer);
    int Direct( double Cotoff,
               double Stopband,
               double Stopband_attenuation,
               int N,
               double *az,double *bz);
    void Filter(vector<double> &data); 

};

#endif