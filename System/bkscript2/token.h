#include <string>
#include <vector>
#ifndef TKN
#define TKN
using namespace std;

enum TknKind {
	P_st, P_ed, C_st, C_ed, B_if, B_else, B_while, B_ed, Comma, Dot,
	Push, Pop, Write, Writeln, Read, Set, T_id, T_literal,
	Eql, Neq, Grt, Sml, Plus, Minus, Mult, Divi, Remain, Other
};

struct Token {
	TknKind kind;
	string text;
	int line;
	Token(TknKind k, string s, int l) { kind = k; text = s; line = l; }
};

extern string source;
extern vector<Token> tokens;

string convert_bk(const char* s);
void get_tokens();

#endif