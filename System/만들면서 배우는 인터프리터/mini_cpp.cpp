#include<iostream>
#include<string>
#include<vector>
#include<stack>
using namespace std;

// gcc mini_cpp.cpp -o t.exe -lstdc++ -finput-charset=UTF-8 -fexec-charset=CP949

void vector_use() {
    vector<int> v;
	vector<int>::iterator p;

	v.push_back(10);
	v.push_back(20);
	v.push_back(30);

	for (p = v.begin(); p != v.end(); p++) cout << *p << " ";
	cout << endl;

	for (int i = 0; i < (int)v.size(); i++) cout << v[i] << " ";
	cout << endl;
}

void stack_use() {
	stack<int> st;
	st.push(10);
	st.push(20);
	st.push(30);

	while (!st.empty()) {
		cout << st.top() << " (" << st.size() << ") ";
		st.pop();
	}
	cout << endl;
}

void error_use() {
	char* p;
	try {
		p = new char[0x7FFFFFFFFFFFFFF0];
	}
	catch (bad_alloc e) { // (exception)
		cout << e.what() << endl;
	}
}

class Person {
private:
	string name;
	int age;
public:
	Person() { name = ""; age = 0; }
	Person(string s) { name = s; age = 0; }
	Person(int n) { name = ""; age = n; }
	Person(string s, int n) { name = s; age = n; }

	void setName(string s);
	void setAge(int n);
	string getName();
	int getAge();
	void disp();
};

void Person::setName(string s) { name = s; }
void Person::setAge(int n) { age = n; if (n < 0) age = 0; }
string Person::getName() { return name; }
int Person::getAge() { return age; }
void Person::disp() { cout << "name : " << name << ", age : " << age << endl; }

int main() {
	vector_use();
	stack_use();
	error_use();

	Person d1("Kim", 10);
	Person d2("Lee", -20);
	d1.disp();
	d2.disp();

	d1.setName("Hong");
	d1.setAge(-50);
	d1.disp();

	cout << "d2 name : " << d2.getName() << ", d2 age : " << d2.getAge() << endl;

	return 0;
}
