#include <stdio.h>
#include "test553b.h"
#include "test553c.h"

int main() {
	Member x;
	x.no = 30;
	x.name[0] = 'B'; x.name[1] = 'K'; x.name[2] = 'a'; x.name[3] = '\0';
	Member y;
	y.no = 48;
	y.name[0] = 'B'; y.name[1] = 'K'; y.name[2] = 'b'; y.name[3] = '\0';
	Member z;
	z.no = 17;
	z.name[0] = 'B'; z.name[1] = 'K'; z.name[2] = 'c'; z.name[3] = '\0';

	Hash1 chunk1;
	Init1(&chunk1, 9);
	Add1(&chunk1, &x);
	Add1(&chunk1, &y);
	Add1(&chunk1, &z);
	Dump1(&chunk1);
	Remove1(&chunk1, &x);
	Dump1(&chunk1);
	Ter1(&chunk1);

	Hash2 chunk2;
	Init2(&chunk2, 9);
	Add2(&chunk2, &x);
	Add2(&chunk2, &y);
	Add2(&chunk2, &z);
	Dump2(&chunk2);
	Remove2(&chunk2, &y);
	Dump2(&chunk2);
	Ter2(&chunk2);

	return 0;
}