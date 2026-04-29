#include <stdio.h>
#include <stdlib.h>

struct NodeA {
	int data;
	struct NodeA* next;
};

struct ListA {
	struct NodeA* head;
	int num;
};

void InitA(struct ListA* list) {
	list->num = 0;
	list->head = NULL;
}

int SeekA(struct ListA* list, int index) {
	if (list->num > index) {
		int i = 0;
		struct NodeA* node = list->head;
		for (i = 0; i < index; i++) {
			node = node->next;
		}
		return node->data;
	}
	else {
		return -1;
	}
}

void InsertA(struct ListA* list, int index, int value) {
	if (list->num < index) {
		index = list->num;
	}
	struct NodeA* temp = malloc(sizeof(struct NodeA));
	temp->data = value;
	int i;
	struct NodeA* node = list->head;

	if (index == 0) {
		temp->next = list->head;
		list->head = temp;
	}
	else if (index == list->num) {
		for (i = 0; i < list->num - 1; i++) {
			node = node->next;
		}
		temp->next = NULL;
		node->next = temp;
	}
	else {
		for (i = 0; i < index - 1; i++) {
			node = node->next;
		}
		temp->next = node->next;
		node->next = temp;
	}

	list->num++;
}

void DelA(struct ListA* list, int index) {
	int i;
	struct NodeA* node = list->head;
	if (list->num > index) {
		if (list->num == 1) {
			list->head = NULL;
			free(node);
		}
		else {
			if (index == 0) {
				list->head = node->next;
				free(node);
			}
			else if (index == list->num - 1) {
				for (i = 0; i < index - 1; i++) {
					node = node->next;
				}
				struct NodeA* temp = node->next;
				node->next = NULL;
				free(temp);
			}
			else {
				for (i = 0; i < index - 1; i++) {
					node = node->next;
				}
				struct NodeA* temp = node->next;
				node->next = temp->next;
				free(temp);
			}
		}

		list->num--;
	}
}

void PrintA(struct ListA* list) {
	struct NodeA* node = list->head;
	printf("<ListA>\n");
	while (node != NULL) {
		printf("%d ", node->data);
		node = node->next;
	}
	printf("\n</ListA>\n");
}

void ClearA(struct ListA* list) {
	struct NodeA* node = list->head;
	struct NodeA* temp;
	for (int i = 0; i < list->num; i++) {
		temp = node;
		node = node->next;
		free(temp);
	}
}

struct ListB {
	int size;
	int max;
	int start;
	int* array;
};

void InitB(struct ListB* list, int max) {
	list->size = 0;
	list->max = max;
	list->start = -1;
	list->array = calloc(max * 2, sizeof(int));
	// 앞 0 ~ max - 1 번은 포인터, 뒤 max ~ 2 * max - 1 번은 데이터
	for (int i = 0; i < max; i++) {
		list->array[i] = -2; // 비어있음은 -2
	}
}

int SeekB(struct ListB* list, int index) {
	if (list->size > index) {
		int temp = list->start;
		for (int i = 0; i < index; i++) {
			temp = list->array[temp];
		}
		return list->array[temp + list->max];
	}
	else {
		return -1;
	}
}

void InsertB(struct ListB* list, int index, int value) {
	if (list->size < index) {
		index = list->size;
	}
	if (list->max > list->size) {
		int new = 0;
		while (list->array[new] != -2) {
			new++;
		}
		list->array[new + list->max] = value;

		if (list->size == 0) {
			list->array[new] = -1;
			list->start = new;
		}
		else {
			if (index == 0) {
				list->array[new] = list->start;
				list->start = new;
			}
			else {
				int temp = list->start;
				for (int i = 0; i < index - 1; i++) {
					temp = list->array[temp];
				}
				list->array[new] = list->array[temp];
				list->array[temp] = new;
			}
		}

		list->size++;
	}
}

void DelB(struct ListB* list, int index) {
	if (list->size > index) {
		if (list->size == 1) {
			list->array[list->start] = -2;
			list->start = -1;
		}
		else {
			if (index == 0) {
				int temp = list->start;
				list->start = list->array[temp];
				list->array[temp] = -2;
			}
			else {
				int temp = list->start;
				for (int i = 0; i < index - 1; i++) {
					temp = list->array[temp];
				}
				int next = list->array[temp];
				list->array[temp] = list->array[next];
				list->array[next] = -2;
			}
		}

		list->size--;
	}
}

void PrintB(struct ListB* list) {
	int temp = list->start;
	printf("<ListB>\n");
	while (temp != -1) {
		printf( "%d ", list->array[list->max + temp] );
		temp = list->array[temp];
	}
	printf("\n</ListB>\n");
}

void ClearB(struct ListB* list) {
	list->max = 0;
	list->size = 0;
	list->start = 0;
	free(list->array);
}

struct ListC {
	int size;
	int max;
	int start;
	int* index;
	int* stack; // 빈 공간을 할당함을 스택으로 관리
	double* data;
};

void InitC(struct ListC* list, int max) {
	list->size = 0;
	list->max = max;
	list->start = -1;
	list->index = malloc( max * sizeof(int) );
	for (int i = 0; i < max; i++) {
		list->index[i] = -1;
	}
	list->stack = malloc(max * sizeof(int));
	for (int i = 0; i < max; i++) {
		list->stack[i] = max - i - 1;
	}
	list->data = malloc(max * sizeof(double));
}

double SeekC(struct ListC* list, int index) {
	if (list->size > index) {
		int temp = list->start;
		for (int i = 0; i < index; i++) {
			temp = list->index[temp];
		}
		return list->data[temp];
	}
	else {
		return 0.0;
	}
}

void InsertC(struct ListC* list, int index, double value) {
	if (list->size < index) {
		index = list->size;
	}
	if (list->max > list->size) {
		int new = list->stack[list->max - list->size - 1];
		list->data[new] = value;

		if (list->size == 0) {
			list->index[new] = -1;
			list->start = new;
		}
		else {
			if (index == 0) {
				list->index[new] = list->start;
				list->start = new;
			}
			else {
				int temp = list->start;
				for (int i = 0; i < index - 1; i++) {
					temp = list->index[temp];
				}
				list->index[new] = list->index[temp];
				list->index[temp] = new;
			}
		}

		list->size++;
	}
}

void DelC(struct ListC* list, int index) {
	if (list->size > index) {
		int deleted;

		if (list->size == 1) {
			deleted = list->start;
			list->index[list->start] = -1;
			list->start = -1;
		}
		else {
			if (index == 0) {
				deleted = list->start;
				list->start = list->index[deleted];
				list->index[deleted] = -1;
			}
			else {
				int temp = list->start;
				for (int i = 0; i < index - 1; i++) {
					temp = list->index[temp];
				}
				deleted = list->index[temp];
				list->index[temp] = list->index[deleted];
				list->index[deleted] = -1;
			}
		}

		list->stack[list->max - list->size] = deleted;

		list->size--;
	}
}

void PrintC(struct ListC* list) {
	int temp = list->start;
	printf("<ListC>\n");
	while (temp != -1) {
		printf("%lf ", list->data[temp]);
		temp = list->index[temp];
	}
	printf("\n</ListC>\n");
}

void ClearC(struct ListC* list) {
	list->size = 0;
	list->max = 0;
	list->start = 0;
	free(list->index);
	free(list->stack);
	free(list->data);
}

struct NodeD {
	int value;
	struct NodeD* prev;
	struct NodeD* next;
};

struct ListD {
	int num;
	struct NodeD* head;
};

void InitD(struct ListD* list) {
	list->num = 0;
	struct NodeD* temp = malloc(sizeof(struct NodeD));
	temp->prev = temp;
	temp->next = temp;
	list->head = temp;
}

int SeekD(struct ListD* list, int index) {
	if (list->num / 2 > index) {
		int i;
		struct NodeD* node = list->head;
		node = node->next; // 더미 다음 실제 노드
		for (i = 0; i < index; i++) {
			node = node->next;
		}
		return node->value;
	}
	else if (list->num > index) {
		int i;
		struct NodeD* node = list->head;
		node = node->prev; // 더미 전 실제 노드
		for (i = list->num - 1; i > index; i--) {
			node = node->prev;
		}
		return node->value;
	}
	else {
		return -1;
	}
}

void InsertD(struct ListD* list, int index, int value) {
	if (list->num < index) {
		index = list->num;
	}

	struct NodeD* new = malloc(sizeof(struct NodeD));
	new->value = value;
	int i;
	struct NodeD* node = list->head;
	struct NodeD* temp;

	if (index == 0) {
		if (list->num == 0) {
			new->prev = node;
			new->next = node;
			node->prev = new;
			node->next = new;
		}
		else {
			new->prev = node;
			new->next = node->next;
			temp = node->next;
			temp->prev = new;
			node->next = new;

		}
	}
	else if (index == list->num) {
		new->prev = node->prev;
		new->next = node;
		temp = node->prev;
		temp->next = new;
		node->prev = new;
	}
	else if (list->num / 2 > index) {
		for (i = -1; i < index; i++) {
			node = node->next;
		}
		new->prev = node->prev;
		new->next = node;
		temp = node->prev;
		temp->next = new;
		node->prev = new;
	}
	else {
		for (i = list->num; i > index; i--) {
			node = node->prev;
		}
		new->prev = node->prev;
		new->next = node;
		temp = node->prev;
		temp->next = new;
		node->prev = new;
	}

	list->num++;
}

void DelD(struct ListD* list, int index) {
	if (list->num > index) {
		int i;
		struct NodeD* node = list->head;
		struct NodeD* temp;

		if (list->num / 2 > index) {
			for (i = -1; i < index; i++) {
				node = node->next;
			}
		}
		else {
			for (i = list->num; i > index; i--) {
				node = node->prev;
			}
		}

		temp = node->prev;
		temp->next = node->next;
		temp = node->next;
		temp->prev = node->prev;
		free(node);

		list->num--;
	}
}

void PrintD(struct ListD* list) {
	struct NodeD* node = list->head;
	node = node->next; // 더미 다음부터
	printf("<ListD>\n");
	for (int i = 0; i < list->num; i++) {
		printf("%d ", node->value);
		node = node->next;
	}
	printf("\n</ListD>\n");
}

void RPrintD(struct ListD* list) {
	struct NodeD* node = list->head;
	node = node->prev; // 더미 전부터
	printf("<RListD>\n");
	for (int i = 0; i < list->num; i++) {
		printf("%d ", node->value);
		node = node->prev;
	}
	printf("\n</RListD>\n");
}

void ClearD(struct ListD* list) {
	struct NodeD* node = list->head;
	struct NodeD* temp;
	for (int i = 0; i < list->num + 1; i++) {
		temp = node;
		node = node->next;
		free(temp);
	}
}

int main() {
	struct ListA lista;
	InitA(&lista);
	PrintA(&lista);
	InsertA(&lista, 0, 20);
	InsertA(&lista, 1, 40);
	InsertA(&lista, 1, 30);
	InsertA(&lista, 0, 10);
	InsertA(&lista, 4, 50);
	PrintA(&lista);
	int i;
	i = SeekA(&lista, 0);
	printf("%d ", i);
	i = SeekA(&lista, 3);
	printf("%d\n", i);
	DelA(&lista, 4);
	DelA(&lista, 0);
	DelA(&lista, 1);
	PrintA(&lista);
	ClearA(&lista);

	struct ListB listb;
	InitB(&listb, 5);
	PrintB(&listb);
	InsertB(&listb, 0, 20);
	InsertB(&listb, 1, 40);
	InsertB(&listb, 1, 30);
	InsertB(&listb, 0, 10);
	InsertB(&listb, 4, 50);
	PrintB(&listb);
	int j;
	j = SeekB(&listb, 0);
	printf("%d ", j);
	j = SeekB(&listb, 3);
	printf("%d\n", j);
	DelB(&listb, 4);
	DelB(&listb, 0);
	DelB(&listb, 1);
	PrintB(&listb);
	ClearB(&listb);

	struct ListC listc;
	InitC(&listc, 5);
	PrintC(&listc);
	InsertC(&listc, 0, 20.0);
	InsertC(&listc, 1, 40.0);
	InsertC(&listc, 1, 30.0);
	InsertC(&listc, 0, 10.0);
	InsertC(&listc, 4, 50.0);
	PrintC(&listc);
	double k;
	k = SeekC(&listc, 0);
	printf("%lf ", k);
	k = SeekC(&listc, 3);
	printf("%lf\n", k);
	DelC(&listc, 4);
	DelC(&listc, 0);
	DelC(&listc, 1);
	PrintC(&listc);
	ClearC(&listc);

	struct ListD listd;
	InitD(&listd);
	PrintD(&listd);
	InsertD(&listd, 0, 20);
	InsertD(&listd, 1, 40);
	InsertD(&listd, 1, 30);
	InsertD(&listd, 0, 10);
	InsertD(&listd, 4, 50);
	PrintD(&listd);
	InsertD(&listd, 2, 25);
	InsertD(&listd, 1, 15);
	InsertD(&listd, 5, 35);
	PrintD(&listd);
	int ijk;
	ijk = SeekD(&listd, 0);
	printf("%d ", ijk);
	ijk = SeekD(&listd, 3);
	printf("%d\n", ijk);
	DelD(&listd, 5);
	DelD(&listd, 0);
	DelD(&listd, 1);
	DelD(&listd, 3);
	DelD(&listd, 1);
	PrintD(&listd);
	ClearD(&listd);

	return 0;
}