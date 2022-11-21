#include "IIR.h"
using namespace std;

IIR_Filter::IIR_Filter() {

}
 
IIR_Filter::~IIR_Filter() {
     
}
 
int IIR_Filter::Ceil(double input)
{
     if(input != (double)((int)input)) return ((int)input) +1;
     else return ((int)input); 
}
 
 
int IIR_Filter::Complex_Multiple(COMPLEX a,COMPLEX b,double *Res_Real,double *Res_Imag)
{
       *(Res_Real) =  (a.Real_part)*(b.Real_part) - (a.Imag_Part)*(b.Imag_Part);
       *(Res_Imag)=  (a.Imag_Part)*(b.Real_part) + (a.Real_part)*(b.Imag_Part);	   
	 return (int)1; 
}
 
int IIR_Filter::Complex_Division(COMPLEX a,COMPLEX b,double *Res_Real,double *Res_Imag)
{
        *(Res_Real) =  ((a.Real_part)*(b.Real_part) + (a.Imag_Part)*(b.Imag_Part))/
			           ((b.Real_part)*(b.Real_part) + (b.Imag_Part)*(b.Imag_Part));
		
	 *(Res_Imag)=  ((a.Real_part)*(b.Imag_Part) - (a.Imag_Part)*(b.Real_part))/
			           ((b.Real_part)*(b.Real_part) + (b.Imag_Part)*(b.Imag_Part));
 
	 return (int)1; 
}
 
double IIR_Filter::Complex_Abs(COMPLEX a)
{
      return (double)(sqrt((a.Real_part)*(a.Real_part) + (a.Imag_Part)*(a.Imag_Part)));
}

double IIR_Filter::IIRFilter  (double *a, int Lenth_a,
                           double *b, int Lenth_b,
                           double Input_Data,
                           double *Memory_Buffer) 
{
    int Count;
    double Output_Data = 0; 
    int Memory_Lenth = 0;
    
    if(Lenth_a >= Lenth_b) Memory_Lenth = Lenth_a;
    else Memory_Lenth = Lenth_b;
    
    Output_Data += (*a) * Input_Data;  //a(0)*x(n)             
    
    for(Count = 1; Count < Lenth_a ;Count++)
    {
        Output_Data -= (*(a + Count)) *
                       (*(Memory_Buffer + (Memory_Lenth - 1) - Count));                                       
    } 
    
    //------------------------save data--------------------------// 
    *(Memory_Buffer + Memory_Lenth - 1) = Output_Data;
    Output_Data = 0;
    //----------------------------------------------------------// 
    
    for(Count = 0; Count < Lenth_b ;Count++)
    {    	
        Output_Data += (*(b + Count)) *
                       (*(Memory_Buffer + (Memory_Lenth - 1) - Count));      
    }
    
    //------------------------move data--------------------------// 
    for(Count = 0 ; Count < Memory_Lenth -1 ; Count++)
    {
    	*(Memory_Buffer + Count) = *(Memory_Buffer + Count + 1);
    }
    *(Memory_Buffer + Memory_Lenth - 1) = 0;
    //-----------------------------------------------------------//
     // printf("MM:%lf\n", *Memory_Buffer);
    return (double)Output_Data; 
}
 
 
int IIR_Filter::Direct( double Cotoff,
                         double Stopband,
                         double Stopband_attenuation,
                         int N,
                         double *az,double *bz)
{
     printf("Wc =  %lf  [rad/sec] \n" ,Cotoff);
     printf("Ws =  %lf  [rad/sec] \n" ,Stopband);
     printf("As  =  %lf  [dB] \n" ,Stopband_attenuation);
     printf("--------------------------------------------------------\n" );
     printf("N:  %d  \n" ,N);
     printf("--------------------------------------------------------\n" );

     COMPLEX poles[N],poles_1,poles_2;
     double dk = 0;
     int k = 0;
	int count = 0,count_1 = 0;;
	
     if((N%2) == 0) dk = 0.5;
     else dk = 0;

     for(k = 0;k <= ((2*N)-1) ; k++)
     {
          poles_1.Real_part = (0.5)*Cotoff*cos((k+dk)*(PI/N));
     poles_1.Imag_Part= (0.5)*Cotoff*sin((k+dk)*(PI/N));	

          poles_2.Real_part = 1 - poles_1.Real_part ;
     poles_2.Imag_Part=   -poles_1.Imag_Part;   

     poles_1.Real_part = poles_1.Real_part + 1;
          poles_1.Real_part = poles_1.Real_part;

          Complex_Division(poles_1,poles_2,
                                   &poles[count].Real_part,
                                   &poles[count].Imag_Part);
     
     if(Complex_Abs(poles[count])<1)
     {
               poles[count].Real_part = -poles[count].Real_part;
          poles[count].Imag_Part= -poles[count].Imag_Part;	 
               count++;
          if (count == N) break;
     }

     } 

     printf("pk =   \n" );   
     for(count = 0;count < N ;count++)
     {
          printf("(%lf) + (%lf i) \n" ,-poles[count].Real_part
                                   ,-poles[count].Imag_Part);
     }
     printf("--------------------------------------------------------\n" );

     COMPLEX Res[N+1],Res_Save[N+1];

     Res[0].Real_part = poles[0].Real_part; 
     Res[0].Imag_Part= poles[0].Imag_Part;

     Res[1].Real_part = 1; 
     Res[1].Imag_Part= 0;


     for(count_1 = 0;count_1 < N-1;count_1++)
     {
     for(count = 0;count <= count_1 + 2;count++)
     {
          if(0 == count)
          {
                    Complex_Multiple(Res[count], poles[count_1+1],
                                        &(Res_Save[count].Real_part),
                                        &(Res_Save[count].Imag_Part));
          }

          else if((count_1 + 2) == count)
          {
                    Res_Save[count].Real_part  += Res[count - 1].Real_part;
               Res_Save[count].Imag_Part += Res[count - 1].Imag_Part;	
          }		  
          else 
          {
                    Complex_Multiple(Res[count], poles[count_1+1],
                                        &(Res_Save[count].Real_part),
                                        &(Res_Save[count].Imag_Part));
               
               Res_Save[count].Real_part  += Res[count - 1].Real_part;
               Res_Save[count].Imag_Part += Res[count - 1].Imag_Part;
          }
     }

     for(count = 0;count <= N;count++)
     {
               Res[count].Real_part = Res_Save[count].Real_part; 
               Res[count].Imag_Part= Res_Save[count].Imag_Part;
          *(az + N - count) = Res[count].Real_part;
     }
     }

     double K_z = 0.0;
	for(count = 0;count <= N;count++)   {K_z += *(az+count);}
	K_z = (K_z/pow ((double)2,N));
	printf("K =  %lf \n" , K_z);
 
	for(count = 0;count <= N;count++)
	{
             Res[count].Real_part = 0;
	     Res[count].Imag_Part= 0;
	     Res_Save[count].Real_part = 0;
	     Res_Save[count].Imag_Part= 0;
	}
 
      COMPLEX zero;
 
      zero.Real_part  =  1;
      zero.Imag_Part =  0;
 
      Res[0].Real_part = 1; 
      Res[0].Imag_Part= 0;
      Res[1].Real_part = 1; 
      Res[1].Imag_Part= 0;
 
      for(count_1 = 0;count_1 < N-1;count_1++)
      {
	     for(count = 0;count <= count_1 + 2;count++)
	     {
	          if(0 == count)
		   {
 	                Complex_Multiple(Res[count], zero,
						           &(Res_Save[count].Real_part),
						           &(Res_Save[count].Imag_Part));
	          }
 
	          else if((count_1 + 2) == count)
	          {
	                 Res_Save[count].Real_part  += Res[count - 1].Real_part;
			    Res_Save[count].Imag_Part += Res[count - 1].Imag_Part;	
	          }		  
		    else 
		    {
 	                 Complex_Multiple(Res[count],zero,
						           &(Res_Save[count].Real_part),
						           &(Res_Save[count].Imag_Part));
				
			    Res_Save[count].Real_part  += Res[count - 1].Real_part;
			    Res_Save[count].Imag_Part += Res[count - 1].Imag_Part;
		    }
	     }
 
	     for(count = 0;count <= N;count++)
	     {
	           Res[count].Real_part = Res_Save[count].Real_part; 
                 Res[count].Imag_Part= Res_Save[count].Imag_Part;
		    *(bz + N - count) = Res[count].Real_part;
	     }
      }
 
	for(count = 0;count <= N;count++)
	{
           *(bz + N - count) = *(bz + N - count) * K_z;
	}
      	//------------------------display---------------------------------//
      printf("bz =  [" );   
      for(count= 0;count <= N ;count++)
      {
           printf("%lf ", *(bz+count));
      }
      printf(" ] \n" );
      printf("az =  [" );   
      for(count= 0;count <= N ;count++)
      {
           printf("%lf ", *(az+count));
      }
      printf(" ] \n" );
      printf("--------------------------------------------------------\n" );
	 
      return (int)1;
}
 


void IIR_Filter::Filter(vector<double> &data) {
     struct DESIGN_SPECIFICATION IIR_Filter;
 
     IIR_Filter.Cotoff   = (double)(PI/20);         //[red]
     IIR_Filter.Stopband = (double)((PI*2)/20);   //[red]
     IIR_Filter.Stopband_attenuation = 30;        //[dB]

     int N;
 
     IIR_Filter.Cotoff = 2 * tan((IIR_Filter.Cotoff)/2);            //[red/sec]
     IIR_Filter.Stopband = 2 * tan((IIR_Filter.Stopband)/2);   //[red/sec]
 
     N = Ceil(0.5*( log10 ( pow (10, IIR_Filter.Stopband_attenuation/10) - 1) / 
	 	              log10 (IIR_Filter.Stopband/IIR_Filter.Cotoff)));

     double az[N+1] , bz[N+1];
     Direct(IIR_Filter.Cotoff,
	         IIR_Filter.Stopband,
	         IIR_Filter.Stopband_attenuation,
               N,
	         az,bz);

     double *Memory_Buffer;
     Memory_Buffer = (double *) malloc(sizeof(double)*(N+1));  
     memset(Memory_Buffer,
               0,
               sizeof(double)*(N+1));

     for(double it : data) {
          it = IIRFilter( az, (N+1),
                         bz, (N+1),
                         it,
                         Memory_Buffer );
     }

     // for(int i = 0; i < data.size(); i++) {
     //      IIRFilter( az, (N+1),
     //                               bz, (N+1),
     //                               data[i],
     //                               Memory_Buffer );
     // }

     for(int i = 0; i < data.size(); i++) {
          double input = data[i];
          double output = IIRFilter( az, (N+1),
                                   bz, (N+1),
                                   input,
                                   Memory_Buffer );
          data[i] = output;
     }
     
}
 
 
 
 
// int main(void)
// {
//      // int count;
 
//      // struct DESIGN_SPECIFICATION IIR_Filter;
 
//      // IIR_Filter.Cotoff   = (double)(PI/25);         //[red]
//      // IIR_Filter.Stopband = (double)((PI*2)/25);   //[red]
//      // IIR_Filter.Stopband_attenuation = 40;        //[dB]
 
//      // int N;
 
//      // IIR_Filter.Cotoff = 2 * tan((IIR_Filter.Cotoff)/2);            //[red/sec]
//      // IIR_Filter.Stopband = 2 * tan((IIR_Filter.Stopband)/2);   //[red/sec]
 
//      // N = Ceil(0.5*( log10 ( pow (10, IIR_Filter.Stopband_attenuation/10) - 1) / 
// 	//  	              log10 (IIR_Filter.Stopband/IIR_Filter.Cotoff)));
 
   
//      // double az[N+1] , bz[N+1];
//      // Direct(IIR_Filter.Cotoff,
// 	//          IIR_Filter.Stopband,
// 	//          IIR_Filter.Stopband_attenuation,
//      //           N,
// 	//          az,bz);
     
//      // double *Memory_Buffer;
//      // Memory_Buffer = (double *) malloc(sizeof(double)*(N+1));  
//      // memset(Memory_Buffer,
//      //           0,
//      //           sizeof(double)*(N+1));

//      IIR_Filter IIR;
 
//      FILE* Input_Data;
//      FILE* Output_Data;
//      FILE* Input_Data_1;
 
//      double Input = 0 ;
//      double Output = 0;
//      double temp1 = 0, temp2 = 0;
	 
//      Input_Data   = fopen("C:/code/code/Magnetic-beacon-positioning/test/uart_test/Hdata.txt","r"); 
//      Output_Data = fopen("output.txt","w"); 

//      int Count = 0;
//      vector<double> test;
 
//      while(1)
//      {
//           if(fscanf(Input_Data, "%lf,%lf,%lf", &temp2,&Input,&temp2) == EOF)  break;
 
//           test.push_back(Input);
//      }

//      IIR.Filter(test);

//      for(auto it : test) {
//           fprintf(Output_Data,"%lf\n",it);
//      }

//      // Input_Data_1   = fopen("Hdata.txt","r"); 

//      // while(1)
//      // {
//      //      if(fscanf(Input_Data_1, "%lf,%lf,%lf", &temp2,&temp2,&Input) == EOF)  break;

//      //      Output = IIRFilter( az, (N+1),
//      //                               bz, (N+1),
//      //                               Input,
//      //                               Memory_Buffer );
          
//      //      fprintf(Output_Data,"%lf\n",Output);
//      // }
	
//      printf("Finish \n" );
	 
//      return (int)0;
// }