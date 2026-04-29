#include <stdio.h>
#include <string.h>
#include "test553b.h"
#define _CRT_SECURE_NO_WARNINGS

int MemberNoCmp(const Member* x, const Member* y) {
	return x->no < y->no ? -1 : x->no > y->no ? 1 : 0;
}

int MemberNameCmp(const Member* x, const Member* y) {
	return strcmp(x->name, y->name);
}

void PrintMember(const Member* x) {
	printf("%d %s", x->no, x->name);
}

void PrintLnMember(const Member* x) {
	printf("%d %s\n", x->no, x->name);
}