#include <vector>
#ifndef CMPL
#define CMPL
using namespace std;

enum CodeKind {
	ADD, SUB, MUL, DIV, REM, EQL, NEQ, GRT, SML, PUSH, POP,
	JMP, JMPF, WRT, WRL, RED, MPUSH, MPOP, CPUSH, INIT
};

// bicalc : pop2, push1
// push A / pop A / cpush N : push addr / pop to addr / push const
// mpush / mpop A : pop&push / pop to addr
// jmp P / jmpf P : direct jump / pop&false jump
// wrt / wrl / red A : pop data / set data
// init N : prepare var

struct CodeF {
	CodeKind kind;
	int data;
	CodeF(CodeKind k, int d) { kind = k; data = d; }
};

extern vector<CodeF> codes;

void C_program();

#endif