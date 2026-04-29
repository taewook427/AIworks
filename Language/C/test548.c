#include <stdio.h>
#include <stdlib.h>
#define swap(type, x, y) do {type t = x; x = y; y = t;} while(0)

void shuffle(int* arr, int n) {
	int i = 0;
	int temp = 11;
	for (i = 0; i < n; i++) {
		temp = (7 * temp + 5) % n;
		arr[i] = temp;
	}
}

void prt(int* arr, int n) {
	int i = 0;
	for (i = 0; i < n; i++) {
		printf( "%d ", arr[i] );
	}
	printf("\n");
}

void bubble(int* arr, int n) {
	printf("========== bubble sort start ==========\n");
	int k = 0; // 정렬 완성 위치
	prt(arr, n);
	while (k < n - 1) {
		int j;
		int last = n - 1; // 마지막 교환 위치
		for (j = n - 1; j > k; j--) {
			if (arr[j - 1] > arr[j]) {
				swap(int, arr[j - 1], arr[j]);
				last = j;
				prt(arr, n);
			}
		}
		k = last;
	}
	prt(arr, n);
	printf("========== bubble sort end ==========\n");
}

void selection(int* arr, int n) {
	printf("========== selection sort start ==========\n");
	int i, j;
	prt(arr, n);
	for (i = 0; i < n - 1; i++) {
		int min = i;
		for (j = i + 1; j < n; j++) {
			if (arr[j] < arr[min]) {
				min = j;
			}
		}
		swap(int, arr[i], arr[min]);
		prt(arr, n);
	}
	printf("========== selection sort end ==========\n");
}

void insertion(int* arr, int n) {
	printf("========== insertion sort start ==========\n");
	prt(arr, n);
	int i, j;
	for (i = 1; i < n; i++) {
		int temp = arr[i];
		for (j = i; j > 0 && arr[j - 1] > temp; j--) {
			arr[j] = arr[j - 1];
		}
		arr[j] = temp;
		prt(arr, n);
	}
	printf("========== insertion sort end ==========\n");
}

void shell(int* arr, int n) {
	printf("========== shell sort start ==========\n");
	prt(arr, n);
	int i, j, h, ht, hl;
	hl = n / 9 + 1;
	int* jump = calloc(hl, sizeof(int));
	jump[0] = 1;
	for (i = 1; i < hl; i++) {
		jump[i] = jump[i - 1] * 3 + 1;
	}
	for (ht = hl - 1; ht >= 0; ht--) {
		h = jump[ht];
		for (i = h; i < n; i++) {
			int temp = arr[i];
			for (j = i - h; j >= 0 && arr[j] > temp; j = j - h) {
				arr[j + h] = arr[j];
			}
			arr[j + h] = temp;
			prt(arr, n);
		}
	}
	printf("========== shell sort end ==========\n");
	free(jump);
}

void quick(int* arr, int st, int ed, int n) {
	int pl = st;
	int pr = ed;
	int piv = arr[(pl + pr) / 2];
	do {
		while (arr[pl] < piv) {
			pl++;
		}
		while (arr[pr] > piv) {
			pr--;
		}
		if (pl <= pr) {
			swap(int, arr[pl], arr[pr]);
			pl++;
			pr--;
		}
		prt(arr, n);
	} while (pl <= pr);
	if (st < pr) {
		quick(arr, st, pr, n);
	}
	if (pl < ed) {
		quick(arr, pl, ed, n);
	}
}

void doquick(int* arr, int n) {
	printf("========== quick sort start ==========\n");
	prt(arr, n);
	quick(arr, 0, n - 1, n);
	printf("========== quick sort end ==========\n");
}

int cmpr(int* x, int* y) {
	if (*x > *y) {
		return -1;
	}
	else if (*x < *y) {
		return 1;
	}
	else {
		return 0;
	}
}

// arr 배열의 정렬 범위 입력, arr에 정렬된 상태 만들기
void merge(int* arr, int st, int ed, int* buffer, int n) {
	if (st < ed) {
		int cntr = (st + ed) / 2;
		merge(arr, st, cntr, buffer, n); // [st]~[cntr] : cntr - st + 1
		merge(arr, cntr + 1, ed, buffer, n); // [cntr + 1] ~ [ed] : ed - cntr
		int i = 0; // 왼쪽 커서
		int j = 0; // 오른쪽 커서
		int k = st; // buffer 쓰기커서
		while (i < cntr - st + 1 && j < ed - cntr) {
			if (arr[st + i] > arr[cntr + 1 + j]) {
				buffer[k] = arr[cntr + 1 + j];
				j++;
				k++;
			}
			else {
				buffer[k] = arr[st + i];
				i++;
				k++;
			}
		}
		while (i < cntr - st + 1) {
			buffer[k] = arr[st + i];
			i++;
			k++;
		}
		while (j < ed - cntr) {
			buffer[k] = arr[cntr + 1 + j];
			j++;
			k++;
		}
		for (k = st; k <= ed; k++) {
			arr[k] = buffer[k];
		}
		prt(arr, n);
	}
}

void domerge(int* arr, int n) {
	printf("========== merge sort start ==========\n");
	prt(arr, n);
	int* buffer = calloc(n, sizeof(int));
	merge(arr, 0, n - 1, buffer, n);
	free(buffer);
	printf("========== merge sort end ==========\n");
}

void heaptify(int* arr, int st, int ed, int n) {
	int temp = arr[st];
	int child;
	int parent;
	for (parent = st; parent < (ed + 1) / 2; parent = child) {
		int cl = parent * 2 + 1;
		int cr = cl + 1;
		child = (cr <= ed && arr[cr] > arr[cl]) ? cr : cl;
		if (temp >= arr[child]) {
			break;
		}
		arr[parent] = arr[child];
	}
	arr[parent] = temp;
	prt(arr, n);
}

void heap(int* arr, int n) {
	printf("========== heap sort start ==========\n");
	prt(arr, n);
	int i;
	for (i = (n - 1) / 2; i >= 0; i--) {
		heaptify(arr, i, n - 1, n);
	}
	for (i = n - 1; i > 0; i--) {
		swap(int, arr[0], arr[i]);
		heaptify(arr, 0, i - 1, n);
	}
	printf("========== heap sort end ==========\n");
}

// 크기 n인 배열일 때 값은 0 ~ n-1이 존재한다고 가정
void counting(int* arr, int n) {
	printf("========== counting sort start ==========\n");
	prt(arr, n);
	int* count = calloc(n, sizeof(int));
	int* temp = calloc(n, sizeof(int));
	int i;
	for (i = 0; i < n; i++) {
		count[arr[i]]++;
		prt(count, n);
	}
	prt(count, n);
	for (i = 1; i < n; i++) {
		count[i] = count[i] + count[i - 1];
		prt(count, n);
	}
	for (i = n - 1; i >= 0; i--) {
		temp[count[arr[i]] - 1] = arr[i];
		count[arr[i]]--;
		prt(temp, n);
	}
	for (i = 0; i < n; i++) {
		arr[i] = temp[i];
	}
	prt(arr, n);
	free(count);
	free(temp);
	printf("========== counting sort end ==========\n");
}

int main() {
	int* arr[18];

	shuffle(arr, 18);
	bubble(arr, 18);

	shuffle(arr, 18);
	selection(arr, 18);

	shuffle(arr, 18);
	insertion(arr, 18);

	shuffle(arr, 18);
	shell(arr, 18);

	shuffle(arr, 18);
	doquick(arr, 18);

	shuffle(arr, 18);
	printf("========== qsort start ==========\n");
	prt(arr, 18);
	qsort(arr, 18, sizeof(int), (int(*)(void*, void*))cmpr);
	prt(arr, 18);
	printf("========== qsort end ==========\n");

	shuffle(arr, 18);
	domerge(arr, 18);

	shuffle(arr, 18);
	heap(arr, 18);

	shuffle(arr, 18);
	counting(arr, 18);

	return 0;
}