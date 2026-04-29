#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int med3(int a, int b, int c) {
	if ( (a >= b && b >= c) || (c >= b && b >= a) ) {
		return b;
	}
	else if ((a >= c && c >= b) || (b >= c && c >= a)) {
		return c;
	}
	else {
		return a;
	}
}

int main(void) {
	int a, b, c;

	printf("세 정수의 중앙값을 구합니다.\n");
	printf("a 입력 : ");  scanf("%d", &a);
	printf("b 입력 : ");  scanf("%d", &b);
	printf("c 입력 : ");  scanf("%d", &c);

	printf( "%d\n", med3(a, b, c) );

	a = 17;
	b = 27;
	int min = a < b ? a : b; // 3항 연산자
	printf("%d\n", min);
	a = 37;
	b = 27;
	min = a < b ? a : b;
	printf("%d\n", min);
}