# KorizenBench

> k-atusa long-term reasoning benchmark

주어진 정적 퍼즐을 해석하여 시각 인식과 장기 추론 능력을 평가한다.
Evaluates image recognition and long-term reasoning ability by interpreting given static puzzles.

이미지 인식, 연속추론(한 가설을 계속 파고들어 답을 구하는 것), 추론관리(잘못된 가설인지 판단하는 것) 능력이 모두 요구된다.
All image interpreting, continuous reasoning (continuously digging into a hypothesis to find an answer) and reasoning management ability (determining whether a hypothesis is incorrect) are required.

과제는 난이도에 따라 7개의 Level A, 5개의 Level B, 3개의 Level C, 3개의 Level D로 구성된다.
The tasks are composed of 7 Level A, 5 Level B, 3 Level C, and 3Level D according to difficulty.

## Test

```
이 과제는 암호(퍼즐)해석을 통해 장기 추론 능력을 평가합니다.
올바른 해석이면 간명한 규칙과 영문 어구로 답이 풀립니다.
풀이에 학부 저학년 수준의 컴퓨터 과학 지식이 필요할 수 있습니다.
별도의 인간 개입과 힌트는 제공되지 않습니다.

This task assesses long-term reasoning skills through cryptography (puzzle) deciphering.
A correct interpretation yields the answer using simple rules and English phrases.
Solving this may require lower-level undergraduate computer science knowledge.
No human intervention or hints will be provided.
```

표준 테스트에서는 별도의 힌트나 인간 개입 없이 문제지 사진만 주어진다. In standard tests, only a picture of the test paper is provided without separate hints or human intervention.

데이터 오염을 막기 위해 해설은 공개되지 않습니다. 필요하다면 비공개로 연락해 주십시오. To prevent data contamination, the commentary will not be made public. If necessary, please contact us privately.

## Comment

ARC-AGI 벤치마크는 순수하게 추론 지능 자체를 측정한다는 면에서 의미있고, ARC-AGI-3은 환경과의 상호작용을 통해 추론과 규칙 모델링을 한다는 점에서 혁신적이다.

하지만 ARC-AGI-3의 규칙은 너무 단조롭다는 생각이 들었다. 최신 모델과 특화 하네스가 벤치마크 과적합을 일으키고 있는게 아닌가 의심이 들었고, 순수 추론력을 측정하는 테스트를 만들고 싶었다.

KorizenBench는 규칙을 발견하기는 어렵지만 규칙 자체는 비교적 단순하다. 또한 모델의 해석과 데이터 자체가 가설의 y/n 여부를 판정하도록 되어있다. (스스로 환각을 일으키지 않고 베이지안 업데이트를 해야 함)

모델은 문제 이미지를 시각적으로 해석하고, 자신이 선택한 가설에 따라 데이터를 긴 추론으로 분석하며, 동시에 무수한 가설들 중 무엇이 맞는 것인지 관리해야 한다.

D 레벨은 나중에 추가되었고 형식이 좀 다릅니다.
