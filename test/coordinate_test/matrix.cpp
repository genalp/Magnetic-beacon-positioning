#include "matrix.h"


Matrix::Matrix() {

}

Matrix::~Matrix() {

}

//创建 h行l列的矩阵，并将初始各值设定为0
vector<vector<double>> Matrix::creatmatrix(int h,int l)
{
	vector<vector<double>> v;
	for (int i = 0; i < h; i++)
	{
		vector<double>v1(l,0);
		v.push_back(v1);
	}
	return v;
}

//矩阵A+矩阵B=矩阵C，并返回
vector<vector<double>> Matrix::matrix_plus(const vector<vector<double>>&A,const vector<vector<double>>&B)
{
	int h=A.size();
	int l=A[0].size();
	vector<vector<double>> C;
	C=creatmatrix( h, l);
 
	for(int i=0;i<h;i++)
	{
		for (int j = 0; j < l; j++)
		{
			C[i][j]=A[i][j]+B[i][j];  
			if (abs(C[i][j])<epsilon)
			{
				C[i][j]=0.0;
			}
		}
	}
	return C;
}

//矩阵A-矩阵B=矩阵C，并返回
vector<vector<double>> Matrix::matrix_minus(const vector<vector<double>>&A,const vector<vector<double>>&B)
{
	int h=A.size();
	int l=A[0].size();
	vector<vector<double>> C;
	C=creatmatrix( h, l);
 
	for(int i=0;i<h;i++)
	{
		for (int j = 0; j < l; j++)
		{
			C[i][j]=A[i][j]-B[i][j];  
			if (abs(C[i][j])<epsilon)
			{
				C[i][j]=0.0;
			}
		}
	}
	return C;
}

//矩阵A*矩阵B=矩阵C，并返回
vector<vector<double>> Matrix::matrix_multiply(const vector<vector<double>>&A,const vector<vector<double>>&B)
{
	int A_h=A.size();
	int A_l=A[0].size();
	int B_h=B.size();
	int B_l=B[0].size();
	if(A_l !=B_h)
	{
		cout<<"两矩阵维数无法相乘"<<endl;
		exit(0);
	}
	vector<vector<double>> C=creatmatrix(A_h,B_l);
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i][j]=0;
			for (int k = 0; k < A_l; k++)
			{
				C[i][j] +=A[i][k]*B[k][j];
			}
			if (abs(C[i][j])<epsilon)
			{
				C[i][j]=0.0;
			}
			//cout<<C[i][j]<<"\t";
		}
		//cout<<endl;
	}
	return C;
}

//矩阵A*num=矩阵B，并返回
vector<vector<double>> Matrix::matrix_multiply_num(const vector<vector<double>>&A,double num)
{
	int A_h=A.size();
	int A_l=A[0].size();
	vector<vector<double>> B=creatmatrix(A_h,A_l);
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			B[i][j]=num*A[i][j];
		}
	}
	return B;
}

//矩阵A与矩阵B上下叠加获得新的矩阵C,并返回
vector<vector<double>> Matrix::matrix_overlaying_below(const vector<vector<double>>&A,const vector<vector<double>>&B)
{
	//判断矩阵的列是否相等
	int A_h=A.size();
	int A_l=A[0].size();
	int B_h=B.size();
	int B_l=B[0].size();
	if (A_l != B_l)
	{
		cout<<"叠加的矩阵列数不相等"<<endl;
		exit(0);
	}
	//创建
	vector<vector<double>> C=creatmatrix(A_h+B_h,A_l);
	//将A传入
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			C[i][j]=A[i][j];
		}
	}
	//将B传入
	for (int i = 0; i < B_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i+A_h][j]=B[i][j];
		}
	}
	return C;
}

//矩阵A与矩阵B左右叠加，获得新的矩阵C，并返回
vector<vector<double>> Matrix::matrix_overlaying_beside(const vector<vector<double>>&A,const vector<vector<double>>&B)
{
	//判断矩阵的列是否相等
	int A_h=A.size();
	int A_l=A[0].size();
	int B_h=B.size();
	int B_l=B[0].size();
	if (A_h != B_h)
	{
		cout<<"叠加的矩阵行数不相等"<<endl;
		exit(0);
	}
	//创建
	vector<vector<double>> C=creatmatrix(A_h,A_l+B_l);
	//将A传入
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			C[i][j]=A[i][j];
		}
	}
	//将B传入
	for (int i = 0; i < B_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i][j+A_l]=B[i][j];
		}
	}
	return C;
}

//输入矩阵A，输出矩阵A的转置矩阵AT
vector<vector<double>> Matrix::matrix_trans(const vector<vector<double>> &A)
{
	vector<vector<double>> AT=creatmatrix(A[0].size(),A.size());
	int h=AT.size();
	int l=AT[0].size();
	for (int i = 0; i <h ; i++)
	{
		for (int j = 0; j < l; j++)
		{
			AT[i][j]=A[j][i];
		}
	}
	return AT;
}

//输入矩阵A,输出矩阵A的逆矩阵inv_A
vector<vector<double>> Matrix::matrix_inverse(const vector<vector<double>> &A)
{
	if (A.size() != A[0].size())
	{
		cout<<"输入矩阵维数不合法"<<endl;
		exit(0);
	}
	int n=A.size();
	vector<vector<double>> inv_A=creatmatrix(n,n);
	vector<vector<double>> L=creatmatrix(n,n);
	vector<vector<double>> U=creatmatrix(n,n);
	vector<vector<double>> inv_L=creatmatrix(n,n);
	vector<vector<double>> inv_U=creatmatrix(n,n);
//LU分解
	//L矩阵对角元素为1
	for (int i = 0; i < n; i++)
	{
		L[i][i] = 1;   
	}
	//U矩阵第一行
	for (int i = 0; i < n; i++)
	{
		U[0][i]=A[0][i];  
	}
	//L矩阵第一列
	for (int i = 1; i < n; i++)
	{
		L[i][0]=1.0*A[i][0]/A[0][0];  
	}
 
	//计算LU上下三角
	for (int i = 1; i < n; i++)
	{
		//计算U（i行j列）
		for (int j = i; j < n; j++)
		{
			double tem = 0;
			for (int k = 0; k < i; k++)
			{
				tem += L[i][k] * U[k][j];
			}
			U[i][j] = A[i][j] - tem;
			if (abs(U[i][j])<epsilon)
			{
				U[i][j]=0.0;
			}
		}
		//计算L（j行i列）
		for (int j = i ; j < n; j++)
		{
			double tem = 0;
			for (int k = 0; k < i; k++)
			{
				tem += L[j][k] * U[k][i];
			}
			L[j][i] = 1.0*(A[j][i] - tem) / U[i][i];
			if (abs(L[i][j])<epsilon)
			{
				L[i][j]=0.0;
			}
		}
 
	}
	//L U剩余位置设为0
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			if(i>j)
			{
				U[i][j]=0.0;
			}
			else if(i<j)
			{
				L[i][j]=0.0;
			}
		}
	}
	//LU求逆
	//求矩阵U的逆 
	for (int i=0;i<n;i++) 
	{
		inv_U[i][i]=1/U[i][i];// U对角元素的值，直接取倒数
		for (int k=i-1;k>=0;k--)
		{
			double s=0;
			for (int j=k+1;j<=i;j++)
			{
				s=s+U[k][j]*inv_U[j][i];
			}
			inv_U[k][i]=-s/U[k][k];//迭代计算，按列倒序依次得到每一个值，
			if (abs(inv_U[k][i])<epsilon)
			{
				inv_U[k][i]=0.0;
			}
		}
	}
	//求矩阵L的逆
	for (int i=0;i<n;i++)  
	{
		inv_L[i][i]=1; //L对角元素的值，直接取倒数，这里为1
		for (int k=i+1;k<n;k++)
		{
			for (int j=i;j<=k-1;j++)
			{
				inv_L[k][i]=inv_L[k][i]-L[k][j]*inv_L[j][i]; 
				if (abs(inv_L[k][i])<epsilon)
				{
					inv_L[k][i]=0.0;
				}
			}
		}
	}
	inv_A=matrix_multiply(inv_U,inv_L);
	return inv_A;
}

void Matrix::show_matrix(const vector<vector<double>> &A)
{
	int h=A.size();
	int l=A[0].size();
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < l; j++)
		{
			cout<<A[i][j]<<"\t";
		}
		cout<<endl;
	}
}