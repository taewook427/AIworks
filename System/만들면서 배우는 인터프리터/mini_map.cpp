#include<iostream>
#include<string>
#include<map>
using namespace std;

// gcc mini_map.cpp -o t.exe -lstdc++ -finput-charset=UTF-8 -fexec-charset=CP949

int main() {
	map<string, int> m;
	m["ccc"] = 20;
	m["aaa"] = 30;
	m["bbb"] = 10;
	if (m.find("bbb") != m.end()) cout << m["bbb"] << endl;
	return 0;
}
