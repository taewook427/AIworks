#ifndef ___Hash0
#define ___Hash0
#include "test553b.h"

typedef struct __node1 {
	Member data;
	struct __node1* next;
} Node1;

typedef struct {
	int size;
	Node1** table;
} Hash1;

int Init1(Hash1* h, int size);

Node1* Sch1(const Hash1* h, const Member* x);

int Add1(Hash1* h, const Member* x);

int Remove1(Hash1* h, const Member* x);

void Dump1(const Hash1* h);

void Clear1(Hash1* h);

void Ter1(Hash1* h);

typedef enum {
	Occupied, Empty, Deleted
} Status;

typedef struct {
	Member data;
	Status stat;
} Bucket;

typedef struct {
	int size;
	Bucket* table;
} Hash2;

int Init2(Hash2* h, int size);

Bucket* Sch2(const Hash2* h, const Member* x);

int Add2(Hash2* h, Member* x);

int Remove2(Hash2* h, Member* x);

void Dump2(const Hash2* h);

void Clear2(Hash2* h);

void Ter2(Hash2* h);

#endif