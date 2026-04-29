# 패키지 설치
install.packages("ggplot2")
install.packages("survival")

# 패키지 사용
library(ggplot2)
library(survival)

# 50대 남성, 50대 여성으로 데이터 분리
data0 <- lung[complete.cases(lung) & lung$sex == 1 & lung$age > 50 & lung$age < 60, ]
data1 <- lung[complete.cases(lung) & lung$sex == 2 & lung$age > 50 & lung$age < 60, ]
data2 <- lung[complete.cases(lung) & lung$age > 50 & lung$age < 60, ]

# 남녀 각 집단의 생존시간에 대한 대푯값
print( summary(data0$time) )
print( summary(data1$time) )
ggplot(data2, aes(x=sex, y=time)) + geom_point(color='black')

# 독립 이표본 t-test
print( t.test(data0$time, data1$time) )
