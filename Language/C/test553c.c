#include <stdio.h>
#include <stdlib.h>
#include "test553b.h"
#include "test553c.h"

static int hash1(int key, int size) {
	return key % size;
}

static void SetNode1(Node1* n, const Member* x, const Member* next) {
	n->data = *x;
	n->next = next;
}

int Init1(Hash1* h, int size) {
	int i;
	h->table = calloc(size, sizeof(Node1*));
	if (h->table == NULL) {
		h->size = 0;
		return 0;
	}
	h->size = size;
	for (i = 0; i < size; i++) {
		h->table[i] = NULL;
	}
	return 1;
}

Node1* Sch1(const Hash1* h, const Member* x) {
	int key = hash1(x->no, h->size);
	Node1* p = h->table[key];

	while (p != NULL) {
		if (p->data.no == x->no) {
			return p;
		}
		p = p->next;
	}

	return NULL;
}

int Add1(Hash1* h, const Member* x) {
	int key = hash1(x->no, h->size);
	Node1* p = h->table[key];
	Node1* temp;

	while (p != NULL) {
		if (p->data.no == x->no) {
			return 1;
		}
		p = p->next;
	}

	temp = calloc(1, sizeof(Node1));
	if (temp == NULL) {
		return 2;
	}

	SetNode1(temp, x, h->table[key]);
	h->table[key] = temp;
	return 0;
}

int Remove1(Hash1* h, const Member* x) {
	int key = hash1(x->no, h->size);
	Node1* p = h->table[key];
	Node1** pp = &h->table[key];
	
	while (p != NULL) {
		if (p->data.no == x->no) {
			*pp = p->next;
			free(p);
			return 0;
		}
		pp = &p->next;
		p = p->next;
	}

	return 1;
}

void Dump1(const Hash1* h) {
	int i;
	for (i = 0; i < h->size; i++) {
		Node1* p = h->table[i];
		printf("%02d ", i);
		while (p != NULL) {
			printf("-> %d : (%s) ", p->data.no, p->data.name);
			p = p->next;
		}
		printf("\n");
	}
	printf("\n");
}

void Clear1(Hash1* h) {
	int i;
	for(i = 0; i < h->size; i++) {
		Node1* p = h->table[i];
		while (p != NULL) {
			Node1* next = p->next;
			free(p);
			p = next;
		}
		h->table[i] = NULL;
	}
}

void Ter1(Hash1* h) {
	Clear1(h);
	free(h->table);
	h->size = 0;
}

static int hash2(int key, int size) {
	return size - key % size - 1;
}

static int rehash(int key, int size) {
	return hash2(key - 1, size);
}

static void SetBucket(Bucket* n, const Member* x, Status stat) {
	n->data = *x;
	n->stat = stat;
}

int Init2(Hash2* h, int size) {
	int i;
	h->table = calloc(size, sizeof(Bucket));
	if (h->table == NULL) {
		h->size = 0;
		return 0;
	}
	h->size = size;
	for (i = 0; i < size; i++) {
		h->table[i].stat = Empty;
	}
	return 1;
}

Bucket* Sch2(const Hash2* h, const Member* x) {
	int i = 0;
	int key = hash2(x->no, h->size);
	Bucket* p = &h->table[key];
	
	for (i = 0; p->stat != Empty && i < h->size; i++) {
		if (p->stat == Occupied && p->data.no == x->no) {
			return p;
		}
		key = rehash(key, h->size);
		p = &h->table[key];
	}

	return NULL;
}

int Add2(Hash2* h, const Member* x) {
	int i;
	int key = hash2(x->no, h->size);
	Bucket* p = &h->table[key];

	if (Sch2(h, x)) {
		return 1;
	}
	for (i = 0; i < h->size; i++) {
		if (p->stat == Empty || p->stat == Deleted) {
			SetBucket(p, x, Occupied);
			return 0;
		}
		key = rehash(key, h->size);
		p = &h->table[key];
	}
	return 2;
}

int Remove2(Hash2* h, const Member* x) {
	Bucket* p = Sch2(h, x);
	if (p == NULL) {
		return 1;
	}
	p->stat = Deleted;
	return 0;
}

void Dump2(const Hash2* h) {
	int i;
	for (i = 0; i < h->size; i++) {
		printf("%02d : ", i);
		switch (h->table[i].stat) {
		case Occupied: printf("%d(%s)\n", h->table[i].data.no, h->table[i].data.name); break;
		case Empty: printf("-- 미등록 --\n"); break;
		case Deleted: printf("-- 삭제 마침 --\n"); break;
		}
	}
	printf("\n");
}

void Clear2(Hash2* h) {
	int i;
	for (i = 0; i < h->size; i++) {
		h->table[i].stat = Empty;
	}
}

void Ter2(Hash2* h) {
	Clear2(h);
	free(h->table);
	h->size = 0;
}