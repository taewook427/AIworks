#include <stdio.h>
#include <stdlib.h>
#include<string.h>

int getstrlen(char* str) {
	int len = 0;
	while (str[len] != '\0') {
		len++;
	}
	return len; // \0 제외한 길이
}

int getstrchr(char* str, char ch) {
	int i = 0;
	while (str[i] != ch) {
		if (str[i] == '\0') {
			return -1;
		}
		i++;
	}
	return i;
}

int BFmatch(char* str, char* pat) {
	int cur0 = 0; // str cursor
	int cur1 = 0; // pat cursor
	while (str[cur0] != '\0' && pat[cur1] != '\0') {
		if (str[cur0] == pat[cur1]) {
			cur0++;
			cur1++;
		}
		else {
			cur0 = cur0 - cur1 + 1;
			cur1 = 0;
		}
	}
	if (pat[cur1] == '\0') {
		return cur0 - cur1;
	}
	else {
		return -1;
	}
}

int KMPmatch(char* str, char* pat) {
	int len0 = strlen(str);
	int len1 = strlen(pat);
	char* fail = calloc(len1, sizeof(char));
	int i;
	for (i = 1; i < len1; i++) {
		char stat = fail[i - 1];
		if (stat == 0 && pat[0] == pat[i]) {
			fail[i] = 1;
		}
		else if (stat != 0 && pat[stat] == pat[i]) {
			fail[i] = stat + 1;
		}
	}
	int cur0 = 0;
	int cur1 = 0;
	while (str[cur0] != '\0' && pat[cur1] != '\0') {
		if (str[cur0] == pat[cur1]) {
			cur0++;
			cur1++;
		}
		else if (cur1 == 0) {
			cur0++;
		}
		else {
			cur1 = fail[cur1];
		}
	}
	free(fail);
	if (pat[cur1] == '\0') {
		return cur0 - cur1;
	}
	else {
		return -1;
	}
}

// str에 존재하는 문자는 ABCDE뿐임을 가정
int BMmatch(char* str, char* pat) {
	int jump[5] = { 0, 0, 0, 0, 0 };
	int cur0;
	int cur1;
	int len0 = strlen(str);
	int len1 = strlen(pat);
	for (cur0 = 0; cur0 < 5; cur0++) {
		jump[cur0] = len1;
	}
	for (cur0 = 0; cur0 < len1 - 1; cur0++) {
		jump[pat[cur0] - 65] = len1 - cur0 - 1;
	}
	// 'A' -> 65, cur0 = len(pat) - 1
	while (cur0 < len0) {
		cur1 = len1 - 1;
		while (str[cur0] == pat[cur1]) {
			if (cur1 == 0) {
				return cur0;
			}
			cur0--;
			cur1--;
		}
		cur0 += (jump[str[cur0] - 65] > len1 - cur1) ? jump[str[cur0] - 65] : len1 - cur1;
	}
	return -1;
}

int main() {
	char a[4];
	a[0] = 'A';
	a[1] = 'B';
	a[2] = '\0';
	a[3] = 'C';
	char b[4] = {'a', 'b', 'c', '\0'};
	char c[3] = "Kk"; // char c[] = "Kk"; -> c[3]
	char* d = "Hello";
	printf("%s %s %s %s\n", a, b, c, d);

	printf( "%d %d %d %d\n", getstrlen(a), getstrlen(b), strlen(c), strlen(d) );
	char* t0 = strchr(b, (int)'c');
	char* t1 = strrchr(d, (int)'l');
	printf("%d %d %d\n", getstrchr(a, 'B'), ((t0 - b) / sizeof(char)), ((t1 - d) / sizeof(char)));

	a[0] = 'A'; a[1] = 'B'; a[2] = 'C', a[3] = '\0';
	b[0] = 'A'; b[1] = 'B'; b[2] = 'G', b[3] = '\0';
	c[0] = 'A'; c[1] = 'K'; c[2] = '\0';
	printf("%d %d\n", strcmp(&a, &c), strncmp(&a, &b, 2));

	char txt[80] = "ABABCBABABCCABABBCAABBCAABAABCDABDCABACBACBADBCABCBBABCAEDBCABCA";
	char fake[80] = "ABABCBABABCCABABBCAABBCAABAABCDABDCABACBACBADBCABCBBABCDADBCABEC";
	char pat[10] = "ABCA";
	printf("%d %d\n", BFmatch(&txt, &pat), BFmatch(&fake, &pat));
	printf("%d %d\n", KMPmatch(&txt, &pat), KMPmatch(&fake, &pat));
	printf("%d %d\n", BMmatch(&txt, &pat), BMmatch(&fake, &pat));
	t0 = strstr(&txt, &pat);
	t1 = strstr(&fake, &pat);
	if (t0 == NULL) {
		t0 = txt - 1;
	}
	if (t1 == NULL) {
		t1 = fake - 1;
	}
	printf("%d %d\n", ((t0 - txt) / sizeof(char)), ((t1 - fake) / sizeof(char)));

	return 0;
}