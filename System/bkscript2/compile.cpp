#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include "token.h"
#include "compile.h"
using namespace std;

struct SymTbl {
    int maxvar;
	vector<map<string, int>> tbl;
    SymTbl() { maxvar = 0;  tbl.clear(); }
	int Find(string name);
	int AddName(string name);
	void AddSeg();
	void DelSeg();
};

int SymTbl::Find(string name) {
    for (auto it = tbl.rbegin(); it != tbl.rend(); ++it) {
        auto map_it = it->find(name);
        if (map_it != it->end()) {
            return map_it->second;
        }
    }
    return -1;
}

int SymTbl::AddName(string name) {
    int addr = 0;
    for (auto i : tbl) {
        addr += i.size();
    }
    if (maxvar < addr + 1) maxvar = addr + 1;
    if (!tbl.empty()) {
        tbl.back()[name] = addr;
    }
    else {
        map<string, int> new_map;
        new_map[name] = addr;
        tbl.push_back(new_map);
    }
    return addr;
}

void SymTbl::AddSeg() {
    tbl.push_back(map<string, int>());
}

void SymTbl::DelSeg() {
    if (!tbl.empty()) {
        tbl.pop_back();
    }
}

int tknpos;
vector<CodeF> codes;
Token tk = Token(Other, "", 0);
SymTbl tbl = SymTbl();

vector<CodeF> C_statement();
void C_comment();
vector<CodeF> C_io();
vector<CodeF> C_stack();
vector<CodeF> C_assign();
vector<CodeF> C_expr();
vector<CodeF> C_term();
vector<CodeF> C_factor();
vector<CodeF> C_if();
vector<CodeF> C_while();
CodeF C_Terminal();

void NextTkn() {
    if (tknpos < tokens.size()) {
        tk = tokens[tknpos++];
    }
    else {
        tk = Token(Other, "", 0);
    }
}

void C_program() {
    tknpos = 0;
    codes.clear();
    NextTkn();
    codes.push_back(CodeF(INIT, 0));
    while (tk.kind != P_st) {
        NextTkn();
        if (tk.kind == Other) {
            cout << "compile.program : invalid token " << tk.text << " line " << tk.line << endl;
            exit(1);
        }
    }
    NextTkn();
    while (tk.kind != P_ed) {
        if (tk.kind == Other) {
            cout << "compile.program : invalid token " << tk.text << " line " << tk.line << endl;
            exit(1);
        }
        else {
            for (auto i : C_statement()) codes.push_back(i);
        }
    }
    codes[0] = CodeF(INIT, tbl.maxvar);
}

vector<CodeF> C_statement() {
    vector<CodeF> out;
    switch (tk.kind)
    {
    case C_st:
        C_comment();
        break;
    case Write: case Writeln: case Read:
        out = C_io();
        break;
    case Push: case Pop:
        out = C_stack();
        break;
    case Set:
        out = C_assign();
        break;
    case B_if:
        out = C_if();
        break;
    case B_while:
        out = C_while();
        break;
    default:
        cout << "compile.program : invalid token " << tk.text << " line " << tk.line << endl;
        exit(1);
    }
    if (tk.kind != Dot) {
        cout << "expected DOT but " << tk.text << " at line " << tk.line << endl;
        exit(1);
    }
    NextTkn();
    return move(out);
}

void C_comment() {
    while (tk.kind != C_ed) {
        NextTkn();
        if (tk.kind == Other) {
            cout << "compile.program : invalid token " << tk.text << " line " << tk.line << endl;
            return;
        }
    }
    NextTkn();
}

vector<CodeF> C_io() {
    vector<CodeF> out;
    switch (tk.kind) {
    case Write:
        NextTkn();
        out = C_expr();
        out.push_back(CodeF(WRT, 0));
        break;
    case Writeln:
        NextTkn();
        out = C_expr();
        out.push_back(CodeF(WRL, 0));
        break;
    case Read:
        NextTkn();
        if (tk.kind == T_id) {
            int addr = tbl.Find(tk.text);
            if (addr == -1) addr = tbl.AddName(tk.text);
            NextTkn();
            out.push_back(CodeF(RED, addr));
        }
        else {
            cout << "expected ID after read but " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
    }
    return move(out);
}

vector<CodeF> C_stack() {
    vector<CodeF> out;
    switch (tk.kind)
    {
    case Push:
        NextTkn();
        out = C_expr();
        out.push_back(CodeF(MPUSH, 0));
        break;
    case Pop:
        NextTkn();
        if (tk.kind == T_id) {
            int addr = tbl.Find(tk.text);
            if (addr == -1) addr = tbl.AddName(tk.text);
            NextTkn();
            out.push_back(CodeF(MPOP, addr));
        }
        else {
            cout << "expected ID after pop but " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
    }
    return move(out);
}

vector<CodeF> C_assign() {
    NextTkn();
    if (tk.kind == T_id) {
        int addr = tbl.Find(tk.text);
        if (addr == -1) addr = tbl.AddName(tk.text);
        NextTkn();
        vector<CodeF> out = C_expr();
        out.push_back(CodeF(POP, addr));
        return move(out);
    }
    else {
        cout << "expected ID after set but " << tk.text << " at line " << tk.line << endl;
        exit(1);
    }
}

vector<CodeF> C_expr() {
    vector<CodeF> out = C_term();
    while (true) {
        switch (tk.kind)
        {
        case Dot: case Comma:
            return move(out);
        case Eql:
            NextTkn();
            for (auto i : C_term()) out.push_back(i);
            out.push_back(CodeF(EQL, 0));
            break;
        case Neq:
            NextTkn();
            for (auto i : C_term()) out.push_back(i);
            out.push_back(CodeF(NEQ, 0));
            break;
        case Grt:
            NextTkn();
            for (auto i : C_term()) out.push_back(i);
            out.push_back(CodeF(GRT, 0));
            break;
        case Sml:
            NextTkn();
            for (auto i : C_term()) out.push_back(i);
            out.push_back(CodeF(SML, 0));
            break;
        default:
            cout << "unexpected expr token : " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
    }
    return move(out);
}

vector<CodeF> C_term() {
    vector<CodeF> out = C_factor();
    while (true) {
        switch (tk.kind)
        {
        case Dot: case Comma:
            return move(out);
        case Plus:
            NextTkn();
            for (auto i : C_factor()) out.push_back(i);
            out.push_back(CodeF(ADD, 0));
            break;
        case Minus:
            NextTkn();
            for (auto i : C_factor()) out.push_back(i);
            out.push_back(CodeF(SUB, 0));
            break;
        case Eql: case Neq: case Grt: case Sml:
            return move(out);
        default:
            cout << "unexpected term token : " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
    }
    return move(out);
}

vector<CodeF> C_factor() {
    vector<CodeF> out;
    out.push_back(C_Terminal());
    while (true) {
        switch (tk.kind)
        {
        case Dot: case Comma:
            return move(out);
        case Mult:
            NextTkn();
            out.push_back(C_Terminal());
            out.push_back(CodeF(MUL, 0));
            break;
        case Divi:
            NextTkn();
            out.push_back(C_Terminal());
            out.push_back(CodeF(DIV, 0));
            break;
        case Remain:
            NextTkn();
            out.push_back(C_Terminal());
            out.push_back(CodeF(REM, 0));
            break;
        case Plus: case Minus: case Eql: case Neq: case Grt: case Sml:
            return move(out);
        default:
            cout << "unexpected factor token : " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
    }
    return move(out);
}

CodeF C_Terminal() {
    if (tk.kind == T_id) {
        int addr = tbl.Find(tk.text);
        if (addr == -1) {
            cout << "undefined var : " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
        NextTkn();
        return CodeF(PUSH, addr);
    }
    else if (tk.kind == T_literal) {
        CodeF temp = CodeF(CPUSH, stoi(tk.text));
        NextTkn();
        return temp;
    }
    else {
        cout << "unexpected factor token : " << tk.text << " at line " << tk.line << endl;
        exit(1);
    }
}

vector<CodeF> C_if() {
    NextTkn();
    vector<CodeF> out = C_expr();
    if (tk.kind != Comma) {
        cout << "expected COMMA after cond but " << tk.text << " at line " << tk.line << endl;
        exit(1);
    }
    NextTkn();
    vector<CodeF> part0, part1;
    tbl.AddSeg();
    while (true) {
        if (tk.kind == B_ed) {
            break;
        }
        else if (tk.kind == B_else) {
            break;
        }
        else if (tk.kind == Other) {
            cout << "unexpected token : " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
        for (auto i : C_statement()) part0.push_back(i);
    }
    tbl.DelSeg();
    if (tk.kind == B_else) {
        NextTkn();
        tbl.AddSeg();
        while (true) {
            if (tk.kind == B_ed) {
                break;
            }
            else if (tk.kind == Other) {
                cout << "unexpected token : " << tk.text << " at line " << tk.line << endl;
                exit(1);
            }
            for (auto i : C_statement()) part1.push_back(i);
        }
        tbl.DelSeg();
    }
    if (part1.size() == 0) {
        out.push_back(CodeF(JMPF, (int)part0.size() + 1));
        for (auto i : part0) out.push_back(i);
    }
    else {
        out.push_back(CodeF(JMPF, (int)part0.size() + 2));
        for (auto i : part0) out.push_back(i);
        out.push_back(CodeF(JMP, (int)part1.size() + 1));
        for (auto i : part1) out.push_back(i);
    }
    NextTkn();
    return move(out);
}

vector<CodeF> C_while() {
    NextTkn();
    vector<CodeF> out = C_expr();
    if (tk.kind != Comma) {
        cout << "expected COMMA after cond but " << tk.text << " at line " << tk.line << endl;
        exit(1);
    }
    NextTkn();
    vector<CodeF> part0;
    tbl.AddSeg();
    while (true) {
        if (tk.kind == B_ed) {
            break;
        }
        else if (tk.kind == Other) {
            cout << "unexpected token : " << tk.text << " at line " << tk.line << endl;
            exit(1);
        }
        for (auto i : C_statement()) part0.push_back(i);
    }
    tbl.DelSeg();
    int back = (int)out.size() + (int)part0.size() + 1;
    out.push_back(CodeF(JMPF, (int)part0.size() + 2));
    for (auto i : part0) out.push_back(i);
    out.push_back(CodeF(JMP, -back));
    NextTkn();
    return move(out);
}