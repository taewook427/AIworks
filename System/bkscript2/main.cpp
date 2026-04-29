#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include "token.h"
#include "compile.h"
#include "execute.h"
using namespace std;

// gcc token.cpp compile.cpp execute.cpp main.cpp -o t.exe -lstdc++

#define STRBUF_SIZE 102400

void readFile(const string& p, char* b);
void printToken(const vector<Token>& t);
void printAsm(const vector<CodeF>& t);

int main(int argc, char* argv[]) {
    if (argc == 1) {
        cout << "Usage : ./~ source [print, token, il, noexe]\n";
        return 1;
    }
    bool prtconv = false, prttkn = false, prtil = false, exef = true;
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "print") {
            prtconv = true;
        }
        else if (arg == "token") {
            prttkn = true;
        }
        else if (arg == "il") {
            prtil = true;
        }
        else if (arg == "noexe") {
            exef = false;
        }
    }

    char* src_temp = new char[STRBUF_SIZE];
    readFile(argv[1], src_temp);
    source = convert_bk(src_temp);
    delete[] src_temp;
    if (prtconv) cout << source << endl;

    get_tokens();
    if (prttkn) printToken(tokens);

    C_program();
    if (prtil) printAsm(codes);

    if (exef) {
        cout << "========== Program Start ==========" << endl;
        execute();
        cout << "==========  Program End  ==========" << endl;
    }
    return 0;
}

void readFile(const string& path, char* buf) {
    ifstream fin;
    fin.open(path, ios::binary);
    if (!fin) {
        cout << "Fail to open file : " << path << endl;
    }
    else {
        fin.read(buf, STRBUF_SIZE);
        fin.close();
        if (strlen(buf) > STRBUF_SIZE) {
            buf[0] = 0;
            cout << "Source text should be smaller than : " << STRBUF_SIZE << endl;
        }
    }
}

void printToken(const vector<Token>& t) {
    for (auto i : t) cout << i.text << " (" << (int)i.kind << ") line " << i.line << endl;
}

void printAsm(const vector<CodeF>& t) {
    for (auto i : t) {
        switch (i.kind)
        {
        case ADD: cout << "add";  break;
        case SUB: cout << "sub"; break;
        case MUL: cout << "mul"; break;
        case DIV: cout << "div"; break;
        case REM: cout << "rem"; break;
        case EQL: cout << "eql"; break;
        case NEQ: cout << "neq"; break;
        case GRT: cout << "grt"; break;
        case SML: cout << "sml"; break;
        case PUSH: cout << "push &" << i.data; break;
        case POP: cout << "pop &" << i.data; break;
        case JMP: if (i.data < 0) cout << "jmp " << i.data; else cout << "jmp +" << i.data; break;
        case JMPF: if (i.data < 0) cout << "jmpf " << i.data; else cout << "jmpf +" << i.data; break;
        case WRT: cout << "wrt"; break;
        case WRL: cout << "wrl"; break;
        case RED: cout << "red &" << i.data; break;
        case MPUSH: cout << "mpush"; break;
        case MPOP: cout << "mpop &" << i.data; break;
        case CPUSH: cout << "cpush $" << i.data; break;
        case INIT: cout << "init $" << i.data; break;
        default: cout << "???";
        }
        cout << endl;
    }
}

/*
#start

read A.
push 0i.
while A>0i,
  push A.
  set A A+1n.
end.

pop A.
while A!=0i,
  write A.
  pop A.
end.
writeln 0i.

writeln 3p%2i-7n*2i+7p/3i.
writeln 0i==0i.
writeln 0i!=0i.
writeln 3i>7p.
writeln 8n<1p.

set A 0i.
set B 27p.
while B!=1i,
  write B.
  set A A+1p.
  if B%2i==0i,
    set B B/2p.
  else
    set B 1p+3i*B.
  end.
  if A%20p==0i,
    writeln 1n.
  end.
end.
writeln B.

set A 1n.
set B 12i.
while B!=0i,
  set A A*B.
  set B B-1p.
end.
writeln A.

#exit
*/