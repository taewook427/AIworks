#include <stdio.h>
#include "test512.h"

int main() {
	Stk s;
	if (SInit(&s, 4) == -1) {
		printf("stack generation fail\n");
		return 1;
	}
	int temp;
	temp = SPush(&s, 2);
	printf("%d,", temp);
	temp = SPush(&s, 14);
	printf("%d,", temp);
	temp = SPush(&s, 0);
	printf("%d,", temp);
	temp = SPush(&s, -27);
	printf("%d,", temp);
	temp = SPush(&s, 3);
	printf("%d,", temp);
	SPop(&s, &temp);
	printf("pop %d\n", temp);
	SPeek(&s, &temp);
	printf("peek %d\n", temp);
	SPrint(&s);
	temp = SSch(&s, 4);
	printf("%d ", temp);
	temp = SCap(&s);
	printf("%d ", temp);
	temp = SSize(&s);
	printf("%d ", temp);
	temp = SIsFull(&s);
	printf("%d ", temp);
	SClear(&s);
	temp = SIsEmpty(&s);
	printf("%d \n", temp);
	STer(&s);

	int a = 0;
	int* b = &a;
	printf("%d %p ", a, b);
	b = *b; // b´Â null pointer ÀÌ´Ù
	// *b = -1;
	printf("%d %p \n", a, b);

	Quu q;
	if (QInit(&q, 4) == -1) {
		printf("queue generation fail\n");
		return 1;
	}
	QEnq(&q, 2);
	QEnq(&q, 4);
	QEnq(&q, 6);
	QEnq(&q, 8);
	QEnq(&q, 10);
	temp = -197;
	QDeq(&q, &temp);
	printf("pop : %d\n", temp);
	QPrint(&q);
	QDeq(&q, &temp);
	QEnq(&q, -14);
	QEnq(&q, 0);
	QPrint(&q);
	temp = QSch(&q, 6);
	printf("loc : %d\n", temp);
	QTer(&q);

	return 0;
}