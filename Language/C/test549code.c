#include <stdio.h>
#include <stdlib.h>
#include "test512.h"

int main() {
	Iset s0, s1, s2;
	Iinit(&s0, 4);
	Iinit(&s1, 4);
	Iinit(&s2, 4);

	Iadd(&s0, 2);
	Iadd(&s0, 6);
	Iadd(&s0, 10);
	Iadd(&s0, 4);
	Iadd(&s0, 8);
	Iremove(&s0, 10);
	printf("%d ", Imember(&s0, 2));
	printf("%d ", Imember(&s0, 8));
	printf("%d ", Icap(&s0));
	printf("%d\n", Isize(&s0));

	Iassign(&s1, &s0);
	Iassign(&s2, &s0);
	printf("%d\n", Iequal(&s1, &s2));

	Iremove(&s1, 6);
	Iremove(&s2, 4);
	s0 = *Iunion(&s0, &s1, &s2);
	Iprint(&s0);
	s0 = *Iinter(&s0, &s1, &s2);
	Iprint(&s0);
	s0 = *Idiffer(&s0, &s1, &s2);
	Iprint(&s0);

	Iter(&s0);
	Iter(&s1);
	Iter(&s2);

	Bset t0 = 0;
	Bset t1 = 0;
	Bset t2 = 0;

	Badd(&t0, 0);
	printf("%llu\n", t0);
	Badd(&t0, 1);
	printf("%llu\n", t0);
	Badd(&t0, 5);
	printf("%llu\n", t0);
	Badd(&t0, 63);
	printf("%llu\n", t0);

	Bremove(&t0, 63);
	Bremove(&t0, 0);
	printf("%llu\n", t0);

	Badd(&t0, 17);
	Badd(&t0, 36);
	Badd(&t0, 59);
	printf("%d %d\n", Bsize(t0), Bbits);

	printf( "%lld %lld %lld ", Bmember(t0, 0), Bmember(t0, 5), Bmember(t0, 59) );

	Bremove(&t0, 36);
	Bremove(&t0, 59);
	t1 = t0;
	t2 = t0;
	printf( "%d\n", (t1 == t2) );

	Badd(&t1, 20);
	Badd(&t1, 21);
	Badd(&t2, 40);
	Badd(&t2, 41);

	t0 = t1 & t2;
	Bprint(t0);
	t0 = t1 | t2;
	Bprint(t0);
	t0 = t1 & ~t2;
	Bprint(t0);

	return 0;
}