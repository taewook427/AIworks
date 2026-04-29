#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "test512.h"

// 초기화 성공시 0 실패시 -1
int SInit(Stk* s, int max) {
	s->ptr = 0;
	s->stk = calloc(max, sizeof(int));
	if (s->stk == NULL) {
		s->max = 0;
		return -1;
	}
	s->max = max;
	return 0;
}

// 푸시 성공시 0 실패시 -1
int SPush(Stk* s, int x) {
	if (s->ptr >= s->max) {
		return -1;
	}
	else {
		s->stk[s->ptr] = x;
		s->ptr++;
		return 0;
	}
}

// 팝 성공시 0 실패시 -1
int SPop(Stk* s, int* x) {
	if (s->ptr <= 0) {
		return -1;
	}
	else {
		*x = s->stk[s->ptr - 1];
		s->ptr--;
		return 0;
	}
}

// 픽 성공시 0 실패시 -1
int SPeek(const Stk* s, int* x) {
	if (s->ptr <= 0) {
		return -1;
	}
	else {
		*x = s->stk[s->ptr - 1];
		return 0;
	}
}

// 스택 비우기
void SClear(Stk* s) {
	s->ptr = 0;
}

// 스택 최대 용량 구하기
int SCap(const Stk* s) {
	return s->max;
}

// 스택 데이터 수 구하기
int SSize(const Stk* s) {
	return s->ptr;
}

// 비어있으면 1, 아니면 0
int SIsEmpty(const Stk* s) {
	return s->ptr <= 0;
}

// 가득 차면 1, 아니면 0
int SIsFull(const Stk* s) {
	return s->ptr >= s->max;
}

// 임의의 값 검색
int SSch(const Stk* s, int x) {
	int i;
	for (i = s-> ptr - 1; i >= 0; i--) {
		if (s->stk[i] == x) {
			return i;
		}
	}
	return -1;
}

// 바닥 -> 꼭대기 순으로 출력
void SPrint(const Stk* s) {
	int i;
	for (i = 0; i < s->ptr; i++) {
		printf( "%d ", s->stk[i] );
	}
	printf("\n");
}

// 스택 삭제
void STer(Stk* s) {
	if (s->stk != NULL) {
		free(s->stk);
	}
	s->max = 0;
	s->ptr = 0;
}

// 초기화 성공시 0 실패시 -1
int QInit(Quu* q, int max) {
	q->ptr = 0;
	q->size = 0;
	q->quu = calloc(max, sizeof(int));
	if (q->quu == NULL) {
		q->max = 0;
		return -1;
	}
	else {
		q->max = max;
		return 0;
	}
}

// 넣기 성공시 0 실패시 -1
int QEnq(Quu* q, int x) {
	if (q->size < q->max) {
		q->quu[(q->ptr + q->size) % q->max] = x;
		q->size = q->size + 1;
		return 0;
	}
	else {
		return -1;
	}
}

// 빼기 성공시 0 실패시 -1
int QDeq(Quu* q, int* x) {
	if (q->size <= 0) {
		return -1;
	}
	else {
		*x = q->quu[q->ptr];
		q->size = q->size - 1;
		q->ptr = (q->ptr + 1) % q->max;
		return 0;
	}
}

// 임의의 값 검색
int QSch(const Quu* q, int x) {
	int i = 0;
	for (i = 0; i < q->size; i++) {
		if (q->quu[(q->ptr + i) % q->max] == x) {
			return (q->ptr + i) % q->max;
		}
	}
	return -1;
}

// 모든 데이터 출력
void QPrint(const Quu* q) {
	int i = 0;
	for (i = 0; i < q->size; i++) {
		printf("%d ", q->quu[(q->ptr + i) % q->max]);
	}
	printf("\n");
}

// 큐 삭제
void QTer(Quu* q) {
	q->max = 0;
	q->ptr = 0;
	q->size = 0;
	if (q->quu != NULL) {
		free(q->quu);
	}
}