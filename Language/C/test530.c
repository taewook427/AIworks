#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

void maxof(const int arr[]) { // const int* arr과 동일
	int n = sizeof(arr) / sizeof(int); // 배열은 포인터, const는 읽기전용
	int max = 0;
	for (int i = 0; i < n; i++) {
		if (arr[i] > max) {
			max = arr[i];
		}
	}
	printf("%d\n", max);
}

int main(void) {
	int mem0[5];
	for (int i = 0; i < 5; i++) {
		printf("%d 번째 항목 : ", i);
		scanf("%d", &mem0[i]);
	}
	for (int i = 0; i < 5; i++) {
		printf("%d 번째 항목 : %d\n", i, mem0[i]);
	}

	int mem1[] = { 0, 0, 0, 0, 0 }; // int mem1[5] = { 0, 0, 0, 0, 0 };
	printf( "%d, %d\n", sizeof(mem1), sizeof(mem1[0]) );

	int* mem2 = calloc(3, sizeof(int)); // int형 3개, 0으로 초기화
	int* mem3 = malloc(3 * sizeof(int)); // 12바이트 공간, 초기화 안함
	if (mem2 == NULL) {
		puts("메모리 할당 실패"); // 권장
	}
	if (mem3 == NULL) {
		puts("메모리 할당 실패");
	}
	mem3[0] = 0;
	mem3[1] = 0;
	mem3[2] = 0;
	for (int i = 0; i < 3; i++) {
		printf( "%d, %d, ", mem2[i], mem3[i] );
	}
	free(mem2); // 필수 !!!!!!!!!!
	free(mem3);

	printf("%d, ", mem3[0]); // 컴파일은 되는데 쓰레기 값

	int* x;
	int y = 1000;
	x = &y;
	*x = 999;
	printf("%d, ", y);

	mem2 = calloc( 5, sizeof(int) );
	mem2[4] = 2048;
	printf("%d, ", mem2 + 3);
	printf("%d, ", mem2 + 4);
	printf("%d, ", mem2 + 5); // C는 막지 않는다
	printf( "%d, ", *(mem2 + 4) );

	// 0은 모든 포인터로 변환 가능, 그 정체는 NULL 포인터
	mem2[0] = mem2;
	printf("%d\n", mem2[0]); // 포인터는 64비트 정수일 것이다.
	free(mem2);

	mem2 = malloc(12);
	mem2[0] = 168;
	mem2[1] = 183;
	mem2[2] = 174;
	mem3[0] = 256; // C는 막지 않는다
	maxof(mem2);
	free(mem2);

	return 0;
}