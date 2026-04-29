#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "test512.h"

// 생성 성공시 0, 실패시 -1
int Iinit(Iset* s, int max) {
	s->num = 0;
	s->set = calloc(max, sizeof(int));
	if (s->set == NULL) {
		s->max = 0;
		return -1;
	}
	else {
		s->max = max;
		return 0;
	}
}

// 포함시 인덱스 반환, 불포함시 -1
int Imember(Iset* s, int n) {
	int i;
	for (i = 0; i < s->num; i++) {
		if (s->set[i] == n) {
			return i;
		}
	}
	return -1;
}

// 원소 첨가
void Iadd(Iset* s, int n) {
	if (s->num < s->max && Imember(s, n) == -1) {
		s->set[s->num] = n;
		s->num++;
	}
}

// 원소 삭제
void Iremove(Iset* s, int n) {
	int idx = Imember(s, n);
	if (idx != -1) {
		s->set[idx] = s->set[s->num - 1];
		s->num--;
	}
}

// 최대 용량
int Icap(Iset* s) {
	return s->max;
}

// 원소수
int Isize(Iset* s) {
	return s->num;
}

// s0 = s1 대입
void Iassign(Iset* s0, Iset* s1) {
	int i;
	int n = (s0->max < s1->num) ? s0->max : s1->num;
	for (i = 0; i < n; i++) {
		s0->set[i] = s1->set[i];
	}
	s0->num = n;
}

// 1 if s0 == s1 else 0
int Iequal(Iset* s0, Iset* s1) {
	int i, j;
	if (Isize(s0) != Isize(s1)) {
		return 0;
	}
	for (i = 0; i < s0->num; i++) {
		for (j = 0; j < s1->num; j++) {
			if (s0->set[i] == s1->set[i]) {
				break;
			}
		}
		if (j == s1->num) {
			return 0;
		}
	}
	return 1;
}

// s0 = s1 합 s2
Iset* Iunion(Iset* s0, Iset* s1, Iset* s2) {
	int i;
	Iassign(s0, s1);
	for (i = 0; i < s2->num; i++) {
		Iadd(s0, s2->set[i]);
	}
	return s0;
}

// s0 = s1 교 s2
Iset* Iinter(Iset* s0, Iset* s1, Iset* s2) {
	int i, j;
	s0->num = 0;
	for (i = 0; i < s1->num; i++) {
		for (j = 0; j < s2->num; j++) {
			if (s1->set[i] == s2->set[j]) {
				Iadd(s0, s1->set[i]);
			}
		}
	}
	return s0;
}

// s0 = s1 - s2
Iset* Idiffer(Iset* s0, Iset* s1, Iset* s2) {
	int i, j;
	s0->num = 0;
	for (i = 0; i < s1->num; i++) {
		for (j = 0; j < s2->num; j++) {
			if (s1->set[i] == s2->set[j]) {
				break;
			}
		}
		if (j == s2->num) {
			Iadd(s0, s1->set[i]);
		}
	}
	return s0;
}

// print(s0)
void Iprint(Iset* s) {
	int i;
	printf("{ ");
	for (i = 0; i < s->num; i++) {
		printf("%d ", s->set[i]);
	}
	printf("}\n");
}

// 집합 삭제
void Iter(Iset* s) {
	if (s->set != NULL) {
		free(s->set);
	}
	s->num = 0;
	s->max = 0;
}

// s에 n이 포함되는지?
long long Bmember(Bset s, int n) {
	return s & Bnum(n);
}

// s에 n추가
void Badd(Bset* s, int n) {
	*s |= Bnum(n);
}

// s에서 n 제거
void Bremove(Bset* s, int n) {
	*s &= ~Bnum(n);
}

// s의 원소수 반환
int Bsize(Bset s) {
	int n = 0;
	while (s != Bnull) {
		s = s & s - 1; // 가장 먼저 나오는 1이 삭제됨 -> 1의 개수번 반복
		n++;
	}
	return n;
}

// s 출력
void Bprint(Bset s) {
	int i;
	printf("{ ");
	for (i = 0; i < Bbits; i++) {
		if ( Bmember(s, i) ) {
			printf("%d ", i);
		}
	}
	printf("}\n");
}