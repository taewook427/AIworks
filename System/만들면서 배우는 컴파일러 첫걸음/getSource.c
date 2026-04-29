#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "getSource.h"


#define MAXLINE 120
#define MAXERROR 30
#define MAXNUM 14
#define TAB 5
#define INSERT_C "#0000FF"
#define DELETE_C "#FF0000"
#define TYPE_C "#00FF00"


static FILE *fpi;
static FILE *fptex;//LateX processing


static char line[MAXLINE]; //1줄 버퍼
static int  lineIndex;     //버퍼 인덱스
static char ch;            //nextChar() 저장


static Token cToken;//LateX processing
static KindT idKind;//LateX processing
static int spaces;  //LateX processing
static int CR;      //LateX processing
static int printed; //LateX processing


static int errorNo = 0;
static char nextChar();
static void printSpaces();
static void printcToken();//LateX processing


struct keyWd {                   //예약어
	char *word;
	KeyId keyId;
};

static struct keyWd KeyWdT[] = { //토큰명
	{"begin", Begin},
	{"end", End},
	{"if", If},
	{"then", Then},
	{"while", While},
	{"do", Do},
	{"return", Ret},
	{"function", Func},
	{"var", Var},
	{"const", Const},
	{"odd", Odd},
	{"write", Write},
	{"writeln",WriteLn},
	{"$dummy1",end_of_KeyWd},
							
	{"+", Plus},
	{"-", Minus},
	{"*", Mult},
	{"/", Div},
	{"(", Lparen},
	{")", Rparen},
	{"=", Equal},
	{"<", Lss},
	{">", Gtr},
	{"<>", NotEq},
	{"<=", LssEq},
	{">=", GtrEq},
	{",", Comma},
	{".", Period},
	{";", Semicolon},
	{":=", Assign},
	{"$dummy2",end_of_KeySym}
};


int isKeyWd(KeyId k)
{
	return (k < end_of_KeyWd);
}


int isKeySym(KeyId k)
{
	if (k < end_of_KeyWd) return 0;
	return (k < end_of_KeySym);
}


static KeyId charClassT[256];


static void initCharClassT()
{
	int i;
	
	//먼저 모든 배열에 other 채움
	for(i=0; i<256; i++)
		charClassT[i] = others;
	
	//ascii로 첨자
	for(i='0'; i<='9';i++)
		charClassT[i] = digit;
	
	for(i='A'; i<='Z'; i++)
	    charClassT[i] = letter;
	
	for(i='a'; i<='z'; i++)
		charClassT[i] = letter;
	
	charClassT['+'] = Plus;
	charClassT['-'] = Minus;
	charClassT['*'] = Mult;
	charClassT['/'] = Div;
	charClassT['('] = Lparen;
	charClassT[')'] = Rparen;
	charClassT['='] = Equal;
	charClassT['<'] = Lss;
	charClassT['>'] = Gtr;
	charClassT[','] = Comma;
	charClassT['.'] = Period;
	charClassT[';'] = Semicolon;
	charClassT[':'] = colon;
}


int openSource(char fileName[])
{		
	if((fpi = fopen(fileName, "r")) == NULL){
		printf("can't open %s\n", fileName);
		return 0;
	}

	
	char fileNameO[30];
	strcpy(fileNameO,fileName);
	strcat(fileNameO,".html");
	if((fptex = fopen(fileNameO, "w")) == NULL){
		
		printf("can't open %s\n", fileNameO);
		return 0;
	}
	
	
	printf("open\n");
	return 1;
}


void closeSource()
{
	fclose(fpi);
	fclose(fptex);
	printf("close\n");
}


void initSource()
{
	lineIndex = -1;
	ch = '\n';
	printed = 1; //LateX processing
	initCharClassT();

	fprintf(fptex, "<HTML>\n<HEAD>\n<TITLE>compiled source program</TITLE>\n</HEAD>\n<BODY>\n<PRE>\n");
	
	/*
	fprintf(fptex,"\\documentstyle[12pt]{article}\n");   
	fprintf(fptex,"\\begin{document}\n");
	fprintf(fptex,"\\fboxsep=0pt\n");
	fprintf(fptex,"\\def\\insert#1{$\\fbox{#1}$}\n");
	fprintf(fptex,"\\def\\delete#1{$\\fboxrule=.5mm\\fbox{#1}$}\n");
	fprintf(fptex,"\\rm\n");
	*/
}

void finalSource() {
	if (cToken.kind == Period) printcToken();
	else errorInsert(Period);
	//fprintf(fptex, "\n\\end{document}\n");
	fprintf(fptex, "\n</PRE>\n</BODY>\n</HTML>\n");
}

void errorNoCheck()
{
	if (errorNo++ > MAXERROR){
		fprintf(fptex, "too many errors\n</PRE>\n</BODY>\n</HTML>\n");
		//fprintf(fptex, "too many errors\n\\end{document}\n"); 
		printf("abort compilation\n");	
		exit (1);
	}
}


void errorType(char *m)
{
	printSpaces();
	fprintf(fptex, "<FONT COLOR=%s>%s</FONT>", TYPE_C, m);
	//fprintf(fptex, "\\(\\stackrel{\\mbox{\\scriptsize %s}}{\\mbox{", m);
	printcToken();
	//fprintf(fptex, "}}\\)");
	errorNoCheck();
}


void errorInsert(KeyId k)
{
	if (k < end_of_KeyWd){
		fprintf(fptex, "<FONT COLOR=%s>%s</FONT>", TYPE_C, KeyWdT[k].word);
		//fprintf(fptex, "\\ \\insert{{\\bf %s}}", KeyWdT[k].word);
	}else{ 
		fprintf(fptex, "<FONT COLOR=%s><b>%s</b></FONT>", TYPE_C, KeyWdT[k].word);
		//fprintf(fptex, "\\ \\insert{$%s$}", KeyWdT[k].word);
	}
	
	errorNoCheck();
}


void errorMissingId()
{
	//fprintf(fptex, "\\insert{Id}");
	fprintf(fptex, "<FONT COLOR=%s>Id</FONT>", TYPE_C);
	errorNoCheck();
}


void errorMissingOp()
{
	//fprintf(fptex, "\\insert{$\\otimes$}");
	fprintf(fptex, "<FONT COLOR=%s>@</FONT>", TYPE_C);
	errorNoCheck();
}


void errorDelete()
{
	int i=(int)cToken.kind;
	printSpaces();
	printed = 1;//LateX processing
	if (i < end_of_KeyWd){
		fprintf(fptex, "<FONT COLOR=%s><b>%s</b></FONT>", DELETE_C, KeyWdT[i].word);
		//fprintf(fptex, "\\delete{{\\bf %s}}", KeyWdT[i].word);
	}else if (i < end_of_KeySym){
		fprintf(fptex, "<FONT COLOR=%s>%s</FONT>", DELETE_C, KeyWdT[i].word);
		//fprintf(fptex, "\\delete{$%s$}", KeyWdT[i].word);
	}else if (i==(int)Id){
		fprintf(fptex, "<FONT COLOR=%s>%s</FONT>", DELETE_C, cToken.u.id);
		//fprintf(fptex, "\\delete{%s}", cToken.u.id);
	}else if (i==(int)Num){
		fprintf(fptex, "<FONT COLOR=%s>%d</FONT>", DELETE_C, cToken.u.value);
		//fprintf(fptex, "\\delete{%d}", cToken.u.value);
	}
}


void errorMessage(char *m)
{
	fprintf(fptex, "<FONT COLOR=%s>%s</FONT>", TYPE_C, m);
	//fprintf(fptex, "$^{%s}$", m);
	errorNoCheck();
}


void errorF(char *m)
{
	errorMessage(m);
	//fprintf(fptex, "fatal errors\n\\end{document}\n");
	fprintf(fptex, "fatal errors\n</PRE>\n</BODY>\n</HTML>\n");
	if (errorNo) printf("total %d errors\n", errorNo);
	printf("abort compilation\n");	
	exit (1);
}


char nextChar()
{	// 첫 실행
	if(lineIndex == -1){
		
		//줄단위 로드
		if(fgets(line, MAXLINE,fpi) != NULL){
		
			lineIndex = 0;
		
		}else{
		
			errorF("end of file\n");
			exit(1);
		}
	}
	
	char ch = line[lineIndex++];
	
	//라인 끝
	if((ch == '\n') || (ch =='\r')){
		
		//인덱스 -1
		lineIndex = -1;
		return '\n';
	}

	return ch;
}

int errorN() {
	return errorNo;
}


Token nextToken()
{
	int i = 0;                    //문자수 계산
	int num;                      //어휘가 상수값
	KeyId cc;                     //어휘 패턴
	Token temp;                   //반환토큰
	char ident[MAXNAME];          //입력버퍼
	memset(ident, '\0', MAXNAME); //널 패딩
	printcToken();
	spaces = 0;   
	CR = 0;        
	
	
	//다음 토큰까지 공백이나 개행
	while(1){
		
		if(ch == ' ' ){
			
			spaces++;
			
		}else if(ch == '\t'){
			
			spaces+= TAB;
		
		//nextChar처음호출 또는 행말
		}else if(ch == '\n'){
			
			spaces = 0;
			CR++;
		
		//공백 개행 아니면 탈출
		}else{
			
			break;
		}
		
		ch = nextChar();
	}
	
	
	//패턴 조사
	switch (cc = charClassT[ch]){
		
		//영문 시작
		case letter:  
			
			do{
				if(i < MAXNAME){
					ident[i] = ch;
					i++;
					ch = nextChar();
				}
			
			}while( charClassT[ch] == letter || charClassT[ch] == digit );
		
			//단어 최대 길이 초과
			if(i >= MAXNAME){
				errorMessage("too long");
				i = MAXNAME - 1;
			}
			
			//널 패딩
			ident[i] = '\0';
			
			//예약어 확인
			for(i=0; i<end_of_KeyWd; i++){
			
				//예약어 포함
				if(strcmp(ident, KeyWdT[i].word) == 0){	
					
					temp.kind = KeyWdT[i].keyId;
					cToken = temp;//LateX processing
					printed = 0;  //LateX processing
					return temp;
				}
			}
			
			//예약어가 아니면 식별자
			temp.kind = Id;
			strcpy(temp.u.id , ident);
			break;
		
		//숫자로 시작
		case digit:  
			
			num = 0;
			do{
				//자리수올림
				num = 10*num + (ch-'0');
				i++;
				ch = nextChar();
			}while(charClassT[ch] == digit);
		
			//최대 숫자길이 초과
			if(i >= MAXNAME){
				errorMessage("too long");
			}
			
			//토큰 타입은 상수
			temp.kind = Num;
		    temp.u.value = num;
			
			break;
		
		//콜론으로 시작
		case colon:
		
			if( (ch = nextChar()) == '='){
				//대입문
				temp.kind = Assign;
				ch = nextChar();
				break;
			
			}else{
				//nul
				temp.kind = nul;
				break;
			}
		
	    // <로 시작
		case Lss:
		
			if( (ch = nextChar()) == '='){
				//<=
				temp.kind = LssEq;
				ch = nextChar();
				break;
				
			}else if(ch == '>'){
				//<>
				temp.kind = NotEq;
				ch = nextChar();
				break;
			
			}else{
				//<
				temp.kind = Lss;
				break;
			}
		
		//>로 시작
		case Gtr:
	
			if((ch = nextChar()) == '=' ){
				//>=
				temp.kind = GtrEq;
				ch = nextChar();
				break;
				
			}else{
				//>
				temp.kind = Gtr;
				break;
			}
		
		default:
		
			temp.kind = cc;
			ch = nextChar();
			break;
	}
	
	cToken =temp;//LateX processing
	printed = 0; //LateX processing
	return temp;
}


Token checkGet(Token t, KeyId k)
{
	//[const] [ident] [=] 다음 토큰 얻기
	if(t.kind == k) return nextToken();
	
	//[const] [ident] [(예약어) or (연산자, 구분기호)] 건너뛰어 다음토큰
	if ((isKeyWd(t.kind) && isKeyWd(k)) || (isKeySym(t.kind) && isKeySym(k))){
		
		errorDelete(); //LateX processing
		errorInsert(k);//LateX processing
		return nextToken();
	}
	
	errorInsert(k);//LateX processing
	
	return t;
}


static void printSpaces()
{
	while (CR-- > 0){
		fprintf(fptex, "\n");
		//fprintf(fptex, "\\ \\par\n");
	}
	
	while (spaces-- > 0){
		fprintf(fptex, " ");
		//fprintf(fptex, "\\ ");
	}
	
	CR = 0; spaces = 0;
}


static void printcToken()
{
	int i=(int)cToken.kind;
	if (printed){
		printed = 0; return;
	}
	printed = 1;
	printSpaces();
	if (i < end_of_KeyWd){
		fprintf(fptex, "<b>%s</b>", KeyWdT[i].word);
		//fprintf(fptex, "{\\bf %s}", KeyWdT[i].word);
	}else if (i < end_of_KeySym){		
		fprintf(fptex, "%s", KeyWdT[i].word);
		//fprintf(fptex, "$%s$", KeyWdT[i].word);
	}else if (i==(int)Id){
		switch (idKind) {
		case varId: 
			fprintf(fptex, "%s", cToken.u.id);
			//fprintf(fptex, "%s", cToken.u.id); 
			return;
		case parId: 
			fprintf(fptex, "<i>%s</i>", cToken.u.id);
			//fprintf(fptex, "{\\sl %s}", cToken.u.id); 
			return;
		case funcId: 
			fprintf(fptex, "<i>%s</i>", cToken.u.id);
			//fprintf(fptex, "{\\it %s}", cToken.u.id); 
			return;
		case constId: 
			fprintf(fptex, "<tt>%s</tt>", cToken.u.id);
			//fprintf(fptex, "{\\sf %s}", cToken.u.id); 
			return;
		}
	}else if (i==(int)Num){
		fprintf(fptex, "%d", cToken.u.value);
		//fprintf(fptex, "%d", cToken.u.value);
	}
}


void setIdKind(KindT k)
{
	idKind = k;
	return;
}