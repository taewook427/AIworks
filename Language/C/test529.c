#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

void tri(int n) {
	for (int i = 1; i < n + 1; i++) {
		for (int j = 1; j < i; j++) {
			putchar('*');
		}
		putchar('\n');
	}
}

int main(void) {
	int n = 0;
	printf("n은 5의 배수 혹은 6의 배수\nn : ");
	scanf("%d", &n);
	if ( (n % 5 == 0) || (n % 2 == 0) && (n % 3 == 0) ) {
		puts("조건에 맞습니다.");
	}

	for (int i = 1; i < 10; i++) {
		for (int j = 1; j < 10; j++) {
			printf("%3d ", i * j);
		}
		printf("\n");
	}

	printf("%d단 직각 삼각형", n);
	tri(n);

	return 0;
}