기존에 만들었던 언어와 변환기는 컴파일러 이론을 제대로 알기 전에 만들어서
비효율적이거나 일반적이지 않은 상황에선 제약이 존재했습니다.

bkscript1(2022.5)은 컴파일러라 부르기 어려운 간단한 변환만을 수행했고,
kscript5(2024.9)는 실용적으로 쓸 수 있는 컴파일러였으나,
토큰화 과정이 현대적이지 않았고 구문 분석에 재귀적 하향 파싱을 사용했습니다.

bkscript2(2025.1)는 컴파일러 이론을 기반으로 문법을 엄밀하게 구성하여 만든 첫 번째 언어입니다.
LL(1) 문법으로 구성하여 책에서 본 이론을 간단히 실습했습니다.

==========

<프로그램> ::= 안녕 얘들아 {<라인>} 오늘 수업 여기까지
<주석> ::= 에헤이 {.} 여러분 이러면 안됩니다
<라인> ::= <주석> '.' | <입출력> '.' | <스택 접근> '.' | <대입문> '.' | <if문> '.' | <while문> '.'

<입출력> ::= 표현력! <식> | 풀어보세요! <식> | 관심받고 싶어? <식별자>
<스택 접근> ::= 자 앉아보세요 <식> | 너 아웃이야 <식별자>
<대입문> ::= 따라오니~ <식별자> <식>

<식별자> ::= {bk | love | 김 | 병 | 관}
<연산자> ::= 더해 | 빼 | 곱해 | 나눠 | 삼분의 일 법칙 | 같아 | 달라 | 커 | 작아
<식> ::= <식별자> {<연산자> <식>} | <리터럴>
<리터럴> ::= {n}점 | {n}월 모의고사 | {n}월 학력평가 | 주옥같은 기출문제 | 교과서 수준 문제

<if문> ::= 여러분 맞습니까? <식> ',' {<라인>} 불광동 | 여러분 맞습니까? <식> ',' 아닌가? {<라인>} 불광동
<while문> ::= 공부안해? <식> ',' {<라인>} 불광동

==========

<program> ::= p_start {<statement>} p_end
<comment> ::= c_start {.} c_end
<statement> ::= <comment> '.' | <io> '.' | <stack> '.' | <assign> '.' | <if> '.' | <while> '.'

<io> ::= write <expr> | writeln <expr> | read id
<stack> ::= push <expr> | pop id
<assign> ::= set id <expr>

<expr> ::= <term> {[== | != | > | <] <expr>}
<term> ::= <factor> {[+ | -] <term>}
<factor> ::= id {[* | / | %] <factor>} | literal

<if> ::= if <expr> ',' {<statement>} end | if <expr> ',' {<statement>} else {<statement>} end
<while> ::= while <expr> ',' {<statement>} end

==========

First(<program>) = {p_start}
First(<comment>) = {c_start}
First(<statement>) = First(<comment>) + First(<io>) + First(<stack>) + First(<assign>) + First(<if>) + First(<while>)
    = {c_start, write, writeln, read, push, pop, set, if, while}

First(<io>) = {write, writeln, read}
First(<stack>) = {push, pop}
First(<assign>) = {set}

First(<expr>) = First(<term>) = First(<factor>) = {id, literal}
First(<term>) = First(<factor>) = {id, literal}
First(<factor>) = {id, literal}

First(<if>) = {if}
First(<while>) = {while}

Follow(<program>) = {$}
Follow(<comment>) = {.}
Follow(<statement>) = {p_end, end, else} + First(<statement>)
    = {p_end, end, else, c_start, write, writeln, read, push, pop, set, if, while}

Follow(<io>) = {.}
Follow(<stack>) = {.}
Follow(<assign>) = {.}

Follow(<expr>) = {+, -, *, /, %, ==, !=, >, <} + Follow(io) + First(<stack>) + First(<assign>)
    = {+, -, *, /, %, ==, !=, >, <, ., ,}
Follow(<term>) = Follow(<expr>) = {+, -, *, /, %, ==, !=, >, <, ., ,}
Follow(<factor>) = Follow(<term>) = {+, -, *, /, %, ==, !=, >, <, ., ,}

Follow(<if>) = {.}
Follow(<while>) = {.}

==========

First 집합이 겹치지 않고, e가 될 수 있는 비단말의 Follow와 다른 First가 겹치지 않으므로 (e 포함 비단말이 없음) 이 문법은 LL(1) 문법이다.
