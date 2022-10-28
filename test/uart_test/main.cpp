#include "WZSerialPort.cpp"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

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
void DataTransfer(string &input, vector<vector<int>> &Hdata) {
    vector<int> OneData;
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
        int Hx = input[start + 3] << 8 | (unsigned char)input[start + 2];    // Hx=((HxH<<8)|HxL)
        int Hy = input[start + 5] << 8 | (unsigned char)input[start + 4];    // Hy=((HyH<<8)|HyL)
        int Hz = input[start + 7] << 8 | (unsigned char)input[start + 6];    // Hz=((HzH<<8)|HzL)
        OneData.push_back(Hx);
        OneData.push_back(Hy);
        OneData.push_back(Hz);
        Hdata.push_back(OneData);   // 记录一组数据
        vector<int>().swap(OneData);// 清空数组
        start += 22;    // 指针移到下一组数据
    }
}

// 数据存储函数
void DataStorage(vector<vector<int>> &Hdata, ofstream &fout) {
    if(!Hdata.size()) {
        return;
    }
    int n = Hdata.size();
    for(int i = 0; i < n; i++) {
        fout << Hdata[i][0] << ",";
        fout << Hdata[i][1] << ",";
        fout << Hdata[i][2] << endl;
    }
}

int main()
{
    string OriginalData;
    vector<vector<int>> Hdata;
    ofstream fout;
    fout.open("C:/code/code/cpp_test/test/uart_test/test.txt");

	WZSerialPort w;
	if (w.open("COM6"))
	{
		// string str = "hello";
		// w.send(str);
        // 取数据
        OriginalData = w.receive();

        // 转换数据
        DataTransfer(OriginalData, Hdata);

        // 存储数据
        DataStorage(Hdata, fout);

        if(Hdata.size() < 200)
            fout << binaryToHex(OriginalData).c_str() << endl;
        // fout << binaryToHex(w.receive()).c_str() << endl;
        fout.close();
		w.close();
	}
    cout << "save successfully!" << endl;
    system("C:\\code\\code\\cpp_test\\test\\uart_test\\test.txt");
    // system("pause");
	return 0;
}