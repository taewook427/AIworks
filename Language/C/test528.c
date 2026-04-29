#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int main(void) {
	int sum = 0;
	int i = 0;
	int n = 0;
	puts("1부터 n까지의 합을 구합니다");
	printf("n : ");
	scanf("%d", &n);
	while (i <= n) {
		sum = sum + i;
		i = i + 1;
	}
	printf("sum : %d\n", sum);

	for (i = 1; i < n + 1; i++) {
		for (int j = 0; j < i; j++) {
			printf("*");
		}
		printf("\n");
	}

	puts("0은 False, 나머지 정수는 True");
	n = 1;
	while (n) {
		printf("n : %d, 정수 입력 : ", n);
		scanf("%d", &n);
	}

	return 0;
}