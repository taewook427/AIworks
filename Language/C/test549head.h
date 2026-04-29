#ifndef ___IBset
#define ___IBset

typedef struct {
	int max;
	int num;
	int* set;
} Iset;

int Iinit(Iset* s, int max); // 초기화

int Imember(Iset* s, int n); // 멤버인가?

void Iadd(Iset* s, int n); // 원소추가

void Iremove(Iset* s, int n); // 원소제거

int Icap(Iset* s); // 추가 가능 원소수

int Isize(Iset* s); // 원소수

void Iassign(Iset* s0, Iset* s1); // 대입, s0 = s1

int Iequal(Iset* s0, Iset* s1); // 동등성 확인

Iset* Iunion(Iset* s0, Iset* s1, Iset* s2); // s0 = s1 합 s2

Iset* Iinter(Iset* s0, Iset* s1, Iset* s2); // s0 = s1 교 s2

Iset* Idiffer(Iset* s0, Iset* s1, Iset* s2); // s0 = s1 - s2

void Iprint(Iset* s); // 출력

void Iter(Iset* s); // 종료

typedef unsigned long long Bset; // 0~63 정수 집합

#define Bnull (Bset)0 // 공집합
#define Bbits 64 // 유효비트수
#define Bnum(n) ((Bset)1 << (n)) // 집합 {n}

long long Bmember(Bset s, int n); // s에 n 포함?

void Badd(Bset* s, int n); // s에 n 넣기

void Bremove(Bset* s, int n); // s에서 n 빼기

int Bsize(Bset s); // s 크기

void Bprint(Bset s); // s 출력

#endif