#include <stdio.h>
#include <stdlib.h>

static int moved = 0;
static int flag0[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
static int flag1[15] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static int flag2[15] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static int pos[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

typedef struct {
	int size;
	int* arr;
} stk;

void init(stk* s) {
	s->size = 0;
	s->arr = calloc(16, sizeof(int));
	if (s->arr == NULL) {
		printf("메모리 할당 실패\n");
	}
}

void push(stk* s, int x) {
	s->arr[s->size] = x;
	s->size = s->size + 1;
}

int pop(stk* s) {
	s->size = s->size - 1;
	return s->arr[s->size];
}

void del(stk* s) {
	s->size = 0;
	free(s->arr);
}

int factorial(int n) {
	if (n < 2) {
		return 1;
	}
	else {
		return factorial(n - 1) * n;
	}
}

int gcd(int x, int y) {
	if (x < y) {
		int temp = x;
		x = y;
		y = temp;
	}
	if (y == 0) {
		return x;
	}
	else {
		return gcd(y, x % y);
	}
}

void recur0(int x) {
	if (x > 1) {
		recur0(x - 1);
	}
	printf("%d ", x);
	if (x > 2) {
		recur0(x - 2);
	}
}

void recur1(int x) {
	stk mem;
	init(&mem);
	push(&mem, x);

	while (mem.size != 0) {
		int temp = pop(&mem);
		if (temp < 0) {
			printf("%d ", -temp);
		}
		else {
			if (temp > 2) {
				push(&mem, temp - 2);
			}
			push(&mem, -temp);
			if (temp > 1) {
				push(&mem, temp - 1);
			}
		}
	}

	del(&mem);
}

void tower(int x, int from, int to) {
	if (x > 1) {
		tower(x - 1, from, 6 - from - to);
	}
	printf("원판 %d : %d -> %d\n", x, from, to);
	moved++;
	if (x > 1) {
		tower(x - 1, 6 - from - to, to);
	}
}

void look4(int a, int b, int c, int d) {
	if (a == 0) {
		look4(1, 0, 0, 0);
		look4(2, 0, 0, 0);
		look4(3, 0, 0, 0);
		look4(4, 0, 0, 0);
	}
	else
	{
		if (b == 0) {
			int i;
			for (i = 1; i < 5; i++) {
				if (i != a) {
					look4(a, i, 0, 0);
				}
			}
		}
		else {
			if (c == 0) {
				int i;
				for (i = 1; i < 5; i++) {
					if (i != a && i != b) {
						look4(a, b, i, 0);
					}
				}
			}
			else {
				d = 10 - a - b - c;
				for (int i = 1; i < a; i++) { printf(" "); }
				printf("0\n");
				for (int i = 1; i < b; i++) { printf(" "); }
				printf("0\n");
				for (int i = 1; i < c; i++) { printf(" "); }
				printf("0\n");
				for (int i = 1; i < d; i++) { printf(" "); }
				printf("0\n\n");
				moved++;
			}
		}
	}
}

void prt() {
	int i;
	for (i = 0; i < 8; i++) {
		printf("%2d", pos[i]);
	}
	printf("\n");
	moved++;
}

void queen8(int i) {
	int j;
	for (j = 0; j < 8; j++) {
		if (!flag0[j] && !flag1[i + j] && !flag2[i - j + 7]) {
			pos[i] = j;
			if (i == 7) {
				prt();
			}
			else {
				flag0[j] = 1;
				flag1[i + j] = 1;
				flag2[i - j + 7] = 1;
				queen8(i + 1);
				flag0[j] = 0;
				flag1[i + j] = 0;
				flag2[i - j + 7] = 0;
			}
		}
	}
}

int main() {
	int temp = factorial(7);
	printf("%d ", temp);
	temp = factorial(5);
	printf("%d\n", temp);

	temp = gcd(16, 24);
	printf("%d ", temp);
	temp = gcd(17, 1);
	printf("%d ", temp);
	temp = gcd(209672, 5405892);
	printf("%d\n", temp);

	recur0(4);
	printf("\n");
	recur1(4);
	printf("\n");

	tower(5, 1, 3);
	printf("%d 회 이동\n", moved);

	moved = 0;
	look4(0, 0, 0, 0);
	printf("%d 종류\n", moved);

	moved = 0;
	queen8(0);
	printf("%d 종류\n", moved);

	return 0;
}