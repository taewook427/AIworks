#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<cstdlib>
#include<cstring>
#include<cctype>
using namespace std;

// gcc mini_token_p.cpp -o t.exe -lstdc++ -finput-charset=UTF-8 -fexec-charset=CP949

enum TknKind {
	Lparen=1, Rparen, Plus, Minus, Multi, Divi, Assign, Comma, DblQ, Equal, NotEq, Less, LessEq, Great, GreatEq,
	If, Else, End, Print, Ident, IntNum, String, Letter, Digit, EofTkn, Others, END_list
};

struct Token {
	TknKind kind;
	string text;
	int intVal;
	Token() { kind = Others; text = ""; intVal = 0; }
	Token(TknKind k, const string& s, int d = 0) { kind = k; text = s; intVal = d; }
};

void initChTyp();
Token nextTkn();
int nextCh();
bool is_ope2(int c1, int c2);
TknKind get_kind(const string& s);

TknKind ctyp[256];
Token token;
ifstream fin;

struct KeyWord {
	const char* keyName;
	TknKind keyKind;
};

KeyWord KeyWdTbl[] = {
	{"if", If}, {"else", Else}, {"end", End}, {"print", Print}, {"(", Lparen}, {")", Rparen}, {"+", Plus}, {"-", Minus}, {"*", Multi}, {"/", Divi},
	{"=", Assign}, {",", Comma}, {"==", Equal}, {"!=", NotEq}, {"<", Less}, {"<=", LessEq}, {">", Great}, {">=", GreatEq}, {"", END_list}
};

int main(int argc, char* argv[]) {
	if (argc == 1) exit(1);
	fin.open(argv[1]); if (!fin) exit(1);

	cout << "text      kind intVal" << endl;
	initChTyp();
	for (token = nextTkn(); token.kind != EofTkn; token = nextTkn()) {
		cout << left << setw(10) << token.text
			<< right << setw(3) << token.kind
			<< " " << token.intVal << endl;
	}
	return 0;
}

void initChTyp() {
	int i;

	for (i = 0; i < 256; i++) { ctyp[i] = Others; }
	for (i = '0'; i <= '9'; i++) { ctyp[i] = Digit; }
	for (i = 'A'; i <= 'Z'; i++) { ctyp[i] = Letter; }
	for (i = 'a'; i <= 'z'; i++) { ctyp[i] = Letter; }

	ctyp['('] = Lparen; ctyp[')'] = Rparen;
	ctyp['<'] = Less; ctyp['>'] = Great;
	ctyp['+'] = Plus; ctyp['-'] = Minus;
	ctyp['*'] =Multi; ctyp['/'] = Divi;
	ctyp['_'] = Letter; ctyp['='] = Assign;
	ctyp[','] = Comma; ctyp['"'] = DblQ;
}

Token nextTkn() {
	TknKind kd;
	int ch0, num = 0;
	static int ch = ' ';
	string txt = "";

	while (isspace(ch)) { ch = nextCh(); }
	if (ch == EOF) return Token(EofTkn, txt);

	switch (ctyp[ch]) {
	case Letter:
		for (; ctyp[ch] == Letter || ctyp[ch] == Digit; ch = nextCh()) { txt += ch; }
		break;
	case Digit:
		for (num = 0; ctyp[ch] == Digit; ch = nextCh()) { num = num * 10 + ch - '0'; }
		return Token(IntNum, txt, num);
	case DblQ:
		for (ch = nextCh(); ch != EOF && ch != '\n' && ch != '"'; ch = nextCh()) { txt += ch; }
		if (ch != '"') { cout << "문자열 리터럴 종료 안 됨\n"; exit(1); }
		ch = nextCh();
		return Token(String, txt);
	default:
		txt += ch; ch0 = ch; ch = nextCh();
		if (is_ope2(ch0, ch)) { txt += ch; ch = nextCh(); }
	}
	kd = get_kind(txt);
	if (kd == Others) { cout << "잘못된 토큰 : " << txt << endl; exit(1); }
	return Token(kd, txt);
}

int nextCh() {
	static int c = 0;
	if (c == EOF) return c;
	if ((c = fin.get()) == EOF) fin.close();
	return c;
}

bool is_ope2(int c1, int c2) {
	char s[] = "    ";
	if (c1 == '\0' || c2 == '\0') return false;
	s[1] = c1; s[2] = c2;
	return strstr(" <= >= == != ", s) != NULL;
}

TknKind get_kind(const string& s) {
	for (int i = 0; KeyWdTbl[i].keyKind != END_list; i++) {
		if (s == KeyWdTbl[i].keyName) return KeyWdTbl[i].keyKind;
	}
	if (ctyp[s[0]] == Letter) return Ident;
	if (ctyp[s[0]] == Digit) return IntNum;
	return Others;
}

/*
if dt <= 100
  dt = (a+200) * b
else
  dt = 300
end
print "dt=", dt
*/
