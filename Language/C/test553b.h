#ifndef ___Member
#define ___Member

typedef struct {
	int no;
	char name[20];
} Member;

#define MEMBER_NO 1
#define MEMBER_NAME 2

int MemberNoCmp(const Member* x, const Member* y);

int MemberNameCmp(const Member* x, const Member* y);

void PrintMember(const Member* x);

void PrintLnMember(const Member* x);

#endif