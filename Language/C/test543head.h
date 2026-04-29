#ifndef ___StkQ
#define ___StkQ

typedef struct {
	int max;
	int ptr;
	int* stk;
} Stk;

int SInit(Stk* s, int max);

int SPush(Stk* s, int x);

int SPop(Stk* s, int *x);

int SPeek(const Stk* s, int *x);

void SClear(Stk* s);

int SCap(const Stk* s);

int SSize(const Stk* s);

int SIsEmpty(const Stk* s);

int SIsFull(const Stk* s);

int SSch(const Stk* s, int x);

void SPrint(const Stk* s);

void STer(Stk* s);

typedef struct {
	int max;
	int size;
	int ptr; // front & rear
	int* quu;
} Quu;

int QInit(Quu* q, int max);

int QEnq(Quu* q, int x);

int QDeq(Quu* q, int* x);

int QSch(const Quu* q, int x);

void QPrint(const Quu* q);

void QTer(Quu* q);

#endif