#include <stdio.h>
#include <stdlib.h>

struct Queue {
	int start;
	int end;
	int max;
	struct Node** array;
};

void InitQ(struct Queue* q, int max) {
	q->start = 0;
	q->end = 0;
	q->max = max;
	q->array = malloc(max * sizeof(struct Node*));
}

void EnQ(struct Queue* q, struct Node* data) {
	q->array[q->end] = data;
	q->end++;
}

struct Node* DeQ(struct Queue* q) {
	q->start++;
	return q->array[q->start - 1];
}

void ClearQ(struct Queue* q) {
	q->start = 0;
	q->end = 0;
	q->max = 0;
	free(q->array);
}

struct Node {
	int value;
	struct Node* left;
	struct Node* right;
};

// 깊이 우선 탐색, 재귀 또는 스택 필요
void DFS(struct Node* root) {
	// 중위 순회
	if (root->left != NULL) {
		DFS(root->left);
	}
	printf("%d ", root->value);
	if (root->right != NULL) {
		DFS(root->right);
	}
}

// 너비 우선 탐색, 큐 필요
void BFS(struct Node* root) {
	struct Queue q;
	InitQ(&q, 1000);
	EnQ(&q, root);

	struct Node* temp;
	while (q.start != q.end) {
		temp = DeQ(&q);
		printf("%d ", temp->value);
		if (temp->left != NULL) {
			EnQ(&q, temp->left);
		}
		if (temp->right != NULL) {
			EnQ(&q, temp->right);
		}
	}

	ClearQ(&q);
}

void Init(struct Node* root, int seed) {
	root->value = seed;
	root->left = NULL;
	root->right = NULL;
}

struct Node* Search(struct Node* root, int value) {
	if (root == NULL) {
		return NULL;
	}
	else {
		int temp = root->value;
		if (temp == value) {
			return root;
		}
		else if (temp > value) {
			return Search(root->left, value);
		}
		else {
			return Search(root->right, value);
		}
	}
}

void Add(struct Node* root, int value) {
	int temp = root->value;
	if (temp > value) {
		if (root->left == NULL) {
			struct node* new = malloc(sizeof(struct Node));
			Init(new, value);
			root->left = new;
		}
		else {
			Add(root->left, value);
		}
	}
	else if (temp < value) {
		if (root->right == NULL) {
			struct node* new = malloc(sizeof(struct Node));
			Init(new, value);
			root->right = new;
		}
		else {
			Add(root->right, value);
		}
	}
}

void Remove(struct Node** root, int value) {
	struct Node* next;
	struct Node* temp;
	struct Node** left;
	struct Node** p = root;

	while (1) {
		if (*p == NULL) {
			break;
		}
		else {
			if ((*p)->value > value) {
				p = &((*p)->left);
			}
			else if ((*p)->value < value) {
				p = &((*p)->right);
			}
			else {
				break;
			}
		}
	}

	if ((*p)->left == NULL) {
		next = (*p)->right;
	}
	else {
		left = &((*p)->left);
		while ((*left)->right != NULL) {
			left = &((*left)->right);
		}
		next = *left;
		*left = (*left)->left;
		next->left = (*p)->left;
		next->right = (*p)->right;
	}

	temp = *p;
	*p = next;
	free(temp);
}

void Clear(struct Node* root) {
	if (root->left != NULL) {
		Clear(root->left);
	}
	if (root->right != NULL) {
		Clear(root->right);
	}
	free(root);
}

int main() {
	struct Node* a = malloc(sizeof(struct Node)); // 해제는 Clear에서 일괄 해제
	Init(a, 64);
	Add(a, 16);
	Add(a, 128);
	Add(a, 256);
	Add(a, 32);
	Add(a, 8);
	/*
	*           64
	*      16        128
	*    8    32         256
	*/
	DFS(a);
	printf("\n");
	BFS(a);
	printf("\n");
	struct Node* i;
	i = Search(a, 64);
	if (i == NULL) {
		printf("NULL ");
	}
	else {
		printf("%d ", i->value);
	}
	i = Search(a, 56);
	if (i == NULL) {
		printf("NULL ");
	}
	else {
		printf("%d ", i->value);
	}
	i = Search(a, 256);
	if (i == NULL) {
		printf("NULL ");
	}
	else {
		printf("%d ", i->value);
	}
	Add(a, 72);
	Add(a, 36);
	Remove(&a, 16);
	Remove(&a, 64);
	Remove(&a, 256);
	/*
	*               64
	*        16            128
	*     8      32     72      256
	*               36
	*
	*               36
	*         8            128
	*            32     72
	*/
	printf("\n");
	DFS(a);
	printf("\n");
	BFS(a);
	printf("\n");
	Clear(a);

	return 0;
}