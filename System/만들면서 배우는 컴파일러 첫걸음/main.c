#include<stdio.h>
#include "getSource.h"
#include "codegen.h"
#include "compile.h"

// gcc codegen.c compile.c getSource.c table.c main.c -o t.exe

int main(int argc, char* argv[])
{
	if (argc < 2) {
		printf("Usage : ./~ source\n");
		return 1;
	}
	
	if(!openSource(argv[1])) return 1;
	if (compile()) execute(); // 컴파일 후 바로 실행
	closeSource();
	
	return 0;
}
