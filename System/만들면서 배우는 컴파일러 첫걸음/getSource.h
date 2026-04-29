#include <stdio.h>
#ifndef TBL
#define TBL
#include "table.h"
#endif

#define MAXNAME 31

typedef enum keys { // 키 종류
	Begin, End, If, Then, While, Do, Ret, Func, Var, Const, Odd, Write, WriteLn, end_of_KeyWd, Plus, Minus, Mult, Div, Lparen, Rparen, Equal, Lss, Gtr,
	NotEq, LssEq, GtrEq, Comma, Period, Semicolon, Assign, end_of_KeySym, Id, Num, nul, end_of_Token, letter, digit, colon, others
} KeyId;

typedef struct token {
	KeyId kind;
	union {
		char id[MAXNAME];
		int value;
	} u;
} Token;

Token nextToken();
Token checkGet(Token t, KeyId k); // t.kind == k 확인

int openSource(char fileName[]);
void closeSource();
void initSource(); // 테이블 초기화
void finalSource(); // 소스 확인

void errorType(char* m); // 자료형 오류
void errorInsert(KeyId k); // 키 오류
void errorMissingId(); // 이름 오류
void errorMissingOp(); // 연산자 오류
void errorDelete(); // 토큰 버리기
void errorMessage(char* m); // 오류 출력
void errorF(char* m); // 오류 출력
int errorN(); // 오류 개수

void setIdKind(KindT k); // 토큰 종류 설정