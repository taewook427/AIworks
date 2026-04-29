#include<stdio.h>
#include "getSource.h"
#ifndef TBL
#define TBL
#include "table.h"
#endif
#include "codegen.h"

#define MINERROR 3
#define FIRSTADDR 2

static Token token; // 다음 토큰

// 컴파일
static void block(int pIndex); // 블록
static void constDecl(); // 상수선언
static void varDecl(); // 변수선언
static void funcDecl(); // 함수선언
static void statement(); // 문장
static void expression(); // 식
static void term(); // 단항
static void factor(); // 식의 인자
static void condition(); // 조건
static int isStBeginKey(Token t); // 토큰이 문장 맨 앞의 키인지 검사

int compile() {
	int i;
	printf("start compilation\n");
	initSource();
	token = nextToken(); // 첫 토큰
	blockBegin(FIRSTADDR); // 새로운 블록
	block(0); // 더미 블록
	finalSource();
	i = errorN();
	if (i != 0) printf("%d errors\n", i);
	listCode(); // 목적 코드 출력
	return i < MINERROR;
}

void block(int pIndex) {
	int backP;
	backP = genCodeV(jmp, 0); // 내부 함수 점프 명령 백패치
	while (1) {
		switch (token.kind)
		{
		case Const:
			token = nextToken();
			constDecl();
			continue;
		case Var:
			token = nextToken();
			varDecl();
			continue;
		case Func:
			token = nextToken();
			funcDecl();
			continue;
		default:
			break;
		}
		break;
	}
	backPatch(backP); // 내부 함수 점프 재배치
	changeV(pIndex, nextCode()); // 함수 시작 주소 수정
	genCodeV(ict, frameL());

	statement(); // 블록의 문장
	genCodeR(); // 리턴 명령 추가
	blockEnd();
}

void constDecl()
{
	Token temp;
	while(1){ // [const] [ident or (?)]↓
		if(token.kind == Id){                   // [const] [ident]↓
			setIdKind(constId);//LateX processing
			temp = token;                   // [const] [ident]↓
			//getSource.c에서
			token = checkGet(nextToken(), Equal);	// [const] [ident] [?]↓
                                                        // [const] [ident] [=] [num]↓
			if(token.kind == Num){
				enterTconst(temp.u.id, token.u.value);
			}else{
				errorType("number");
			}
			
			token = nextToken();            // [const] [ident] [=] [num] [ (,) or (;) or (num) or (?) ]↓
                                                        
		}else{                                  // [const] [?]↓
			errorMissingId();
		}


                                                        // [const] [ident] [=] [num] [(,) or (num) or (;) or (?)]↓
		if(token.kind != Comma){                // [const] [ident] [=] [num] [       (num) or (;) or (?)]↓
			
			if(token.kind == Id){           // [const] [ident] [=] [num] [num]↓
				errorInsert(Comma);
				continue;
			}else{                          // [const] [ident] [=] [num] [(;) or (?)]↓
				break;
			}
		}else{                                  // [const] [ident] [=] [num] [,]↓  
			token = nextToken();            // [const] [ident] [=] [num] [,] [?]↓
		}
	}
	
                                                        // [const] [ident] [=] [num] [(;) or (?)]↓
	token = checkGet(token, Semicolon);             // [const] [ident] [=] [num] [;] [?] ↓
                                                        
}

void varDecl()
{
	while(1){
                                                       // [var] [?]↓
		if(token.kind == Id){                  // [var] [ident]↓
			setIdKind(varId);//LateX processing
			enterTvar(token.u.id);
			token = nextToken();           // [var] [ident] [?]↓                                 
		}else{
			errorMissingId();
		}		

                 
                                                       // [var] [ident] [(,) or (ident) or (;) or (?)]↓
		if(token.kind != Comma){               // [var] [ident] [       (ident) or (;) or (?)]↓	
			if(token.kind == Id){          // [var] [ident] [ident]↓
				errorInsert(Comma);
				continue;	
			}else{                         // [var] [ident] [ (;) or (?)]↓
				break;
			}	
		}else{                                 // [var] [ident] [,]↓
			token = nextToken();	       // [var] [ident] [,] [?]↓
		}	
	}
                                                       // [var] [ident] [ (;) or (?)]↓
	token = checkGet(token, Semicolon);
                                                       // [var] [ident] [;] [?]↓
}


void funcDecl()
{
	int fIndex;
	
	if (token.kind == Id){                                //[function] [ident]↓
		setIdKind(funcId);//LateX processing
		fIndex = enterTfunc(token.u.id, nextCode());//codegen.c에서
		token = checkGet(nextToken(), Lparen);        //[function] [ident] [?]↓
                                                              //[function] [ident] [(] [(ident) or (?)]↓
		blockBegin(FIRSTADDR);
														
		while(1){                                     //[function] [ident] [(] [(ident) or (?)]↓
			
                        if(token.kind == Id){                 //[function] [ident] [(] [ident]↓
				setIdKind(parId);//LateX processing
				enterTpar(token.u.id);
				token = nextToken();          //[function] [ident] [(] [ident] [(,) or (ident) or (?)]↓
			}else{                                //[function] [ident] [(] [?]↓
				
				break;
			}


                                                              //[function] [ident] [(] [ident] [(,) or (ident) or (?)]↓
			if(token.kind != Comma){              //[function] [ident] [(] [ident] [       (ident) or (?)]↓
				
				if(token.kind == Id){         //[function] [ident] [(] [ident] [ident]↓
					errorInsert(Comma);   //[function] [ident] [(] [ident] [,]↓
					continue;
				}else{                        //[function] [ident] [(] [ident] [?]↓
					break;
				}	
			}else{                                //[function] [ident] [(] [ident] [,]↓
				token = nextToken();          //[function] [ident] [(] [ident] [,] [?]↓	                              
			}
		}
                                                              //[function] [ident] [(] ... [?]↓
		token = checkGet(token, Rparen);              //[function] [ident] [(] ... [)] [(;) or (?)]↓
                endpar();
		
		if(token.kind == Semicolon){                  //[function] [ident] [(] ... [)] [;]↓	
			errorDelete();
			token = nextToken();   
		}
                                                              //[function] [ident] [(] ... [)] [?]↓
		block(fIndex);                                
                                                              //[function] [ident] [(] ... [)]〈block〉[?]↓
		token = checkGet(token, Semicolon);           //[function] [ident] [(] ... [)]〈block〉[;]↓                                                              


	}else{
		errorMissingId();
	}
}

void statement()
{
	int tIndex;
	KindT k;
        int backP,backP2;
		
	while(1){
		
		switch(token.kind){
					
		case Id:	                                            //[ident] 
			tIndex = searchT(token.u.id,varId);
			setIdKind(k=kindT(tIndex));
			if(k != varId && k != parId) errorType("var/par");
			
			token = checkGet(nextToken(), Assign);	            //[ident] [:=]
									    //[ident] [:=] [?]
			expression();                                       //[ident] [:=]〈expression〉
			genCodeT(sto, tIndex);
			
			return;
		
		case Begin:                                            //[begin] 	
			token = nextToken();
			                                               //[begin] [?]
			while(1){
				
				statement();
									
				while(1){			       //[begin]〈statement〉[(;) or (end) or (begin) or (?)]
					
					if(token.kind == Semicolon){   //[begin]〈statement〉[;]	
						token = nextToken();   //[begin]〈statement〉[;] [?]
						break;
					}
					
					if(token.kind == End){         //[begin]〈statement [end]	
						token = nextToken();   //[begin]〈statement〉[end] [?]
						return;
					}
					
					if(isStBeginKey(token)){       //[begin]〈statement〉[begin}
						errorInsert(Semicolon);//[begin]〈statement〉[?]
						break;
					}				
								       //[begin]〈statement〉[?]
					errorDelete();		       //[begin]〈statement〉
					token = nextToken();           //[begin]〈statement〉[?]
				}
			}
						
		case End:                                //[begin]〈statement〉[;]...〈statement〉[end]		
			return;

		case If:                                 //[if]			
			token = nextToken();             //[if] [?]
			condition();                     //[if]〈condition〉
			token = checkGet(token, Then);   //[if]〈condition〉[then]
			backP = genCodeV(jpc, 0);
			statement();                     //[if]〈condition〉[then]〈statement〉
			backPatch(backP);
			return;
		
		case While:                              //[while]			
			token  = nextToken();            //[while] [?]
			backP2 = nextCode();
			condition();                     //[while]〈condition〉
			token = checkGet(token, Do);     //[while]〈condition〉[do]
			backP = genCodeV(jpc, 0);
			statement();                     //[while]〈condition〉[do]〈statement〉
			backP = genCodeV(jpc, backP2);
			backPatch(backP);
			return;
						
		case Ret:                                //[return]
			token = nextToken();             //[return] [?]
			expression();                    //[return]〈expression〉
			genCodeR();
			return;
			
		case Write:                              //[write]
			token = nextToken();             //[write] [?]
			expression();                    //[write]〈expression〉
			genCodeO(wrt);
			return;
					
		case WriteLn:                            //[writeln]
			token = nextToken();
			genCodeO(wrl);
			return;
						
		case Period:                             //[.]
			return;

		case Semicolon:                          //[;]		
			return;		

	    default:
			errorDelete();
			token = nextToken();
			continue;
		}	
	}
}


int isStBeginKey(Token t)
{
	switch (t.kind){
	case If: 
	case Begin: 
	case Ret:
	case While: 
	case Write: 
	case WriteLn:		
		return 1;
	default:
		return 0;
	}
}

void expression()
{
	KeyId k;
	k = token.kind;
	                                    //[(+) or (-) or (e)]
	if (k==Plus || k==Minus){           //[(+) or (-)]
	                                  
		token = nextToken();        //[(+) or (-)] [?]
		term();                     //[(+) or (-)]〈term〉
		if (k==Minus) genCodeO(neg);
	}else{                              //[e]
		term();                     //[e]〈term〉
	}
		                            //[(+) or (-) or (e)]〈term〉
	
	k = token.kind;
	
	while (k==Plus || k==Minus){        //[(+) or (-) or (e)]〈term〉[(+) or (-)]
		
		token = nextToken();        //[(+) or (-) or (e)]〈term〉[(+) or (-)] [?]
		term();                     //[(+) or (-) or (e)]〈term〉[(+) or (-)]〈term〉
			
		if (k==Minus)
			genCodeO(sub);
		else
			genCodeO(add);
			
		k = token.kind;
	}
}


void term()
{
	KeyId k;
	
	factor();                   //〈factor〉
	
	k = token.kind;
	
	while (k==Mult || k==Div){  //〈factor〉[(*) or (/)]
		
		token = nextToken();//〈factor〉[(*) or (/)] [?]
		
		factor();           //〈factor〉[(*) or (/)]〈factor〉
		
		if (k==Mult)
			genCodeO(mul);
		else
			genCodeO(div);
		
		k = token.kind;
	}
}

void factor()
{
	int tIndex, i;
	KeyId k;
	
	if (token.kind==Id){

		tIndex = searchT(token.u.id, varId);
		setIdKind(k=kindT(tIndex));//LateX processing
		
		switch (k) {
		
		//변수명, 인자명
		case varId: case parId:            //[ident]
			genCodeT(lod, tIndex);
			token = nextToken(); 
			break;
		
		//상수명
		case constId:                      //[number]
			genCodeV(lit, val(tIndex));
			token = nextToken(); 
			break;
		
		//함수호출
		case funcId:                                                    //[ident] [(]〈expression〉[)] or [ident] [(]〈expression〉[,]...〈expression〉[)] or [ident] [(] [)]
			
			token = nextToken();
			
			if (token.kind==Lparen){                                //[ident] [(]
				
				//i는 인수 개수
				i=0; 					
				
				token = nextToken();                            //[ident] [(] [?]
				
				if (token.kind != Rparen) {                     //[ident] [(] [?]
					
  					for (; ; ) {
						
						expression();                   //[ident] [(]〈expression〉[?]
						i++;	
						
						if (token.kind==Comma){         //[ident] [(]〈expression〉[,]
							token = nextToken();    //[ident] [(]〈expression〉[,] [?]
							continue;
						}
							
						token = checkGet(token, Rparen);//[ident] [(]〈expression〉[)]
						break;
					}
					
				} else{                                         //[ident] [(] [)]
					
					token = nextToken();
					if (pars(tIndex) != i) errorMessage("\\#par");	
				}
					
				
			
			}else{
				errorInsert(Lparen);
				errorInsert(Rparen);
			}
			
			genCodeT(cal, tIndex);
			break;
		}




	}else if (token.kind==Num){              //[상수]
		
		genCodeV(lit, token.u.value);
		token = nextToken();
	
	}else if (token.kind==Lparen){           //[(]
	
		token = nextToken();
		expression();                    //[(]〈expression
		token = checkGet(token, Rparen); //[(]〈expression〉[)]
	}
		
	//인자이면 오류
	switch (token.kind){
	case Id: case Num: case Lparen:
		errorMissingOp();
		factor();
	default:
		return;
	}	
}

void condition()					
{
	KeyId k;
	if (token.kind==Odd){                  //[odd]
	
		token = nextToken();           //[odd] [?]
		expression();                  //[odd]〈expression〉
		genCodeO(odd);
	
	}else{                                 //[?]
		expression();                  //〈expression〉
		k = token.kind;
		
		switch(k){                     //〈expression〉[(=) or (<>) or (<) or (>) or (<=) or (>=)]
		case Equal: case Lss: case Gtr:
		case NotEq: case LssEq: case GtrEq:
			break;
		default:
			errorType("rel-op");
			break;
		}
		
		token = nextToken();           //〈expression〉[(=) or (<>) or (<) or (>) or (<=) or (>=)] [?]
		expression();                  //〈expression〉[(=) or (<>) or (<) or (>) or (<=) or (>=)]〈expression〉
		switch(k){
		case Equal:	genCodeO(eq);   break;
		case Lss:	genCodeO(ls);   break;
		case Gtr:	genCodeO(gr);   break;
		case NotEq:	genCodeO(neq);  break;
		case LssEq:	genCodeO(lseq); break;
		case GtrEq:	genCodeO(greq); break;
		}
	}
}