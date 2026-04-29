# 패키지 설치
install.packages("ggplot2")
install.packages("survival")

# 패키지 사용
library(ggplot2)
library(survival)

# 기본 폐암 생존 데이터 불러오고 karno 점수가 100점인 항목을 나이순 정렬
data <- lung[complete.cases(lung) & lung$ph.karno == 100, ]
data <- data[ order(data$age), ]

# 성별에 따른 색깔, 추적 종료 원인에 따른 표지 추가
data$color <- ifelse(data$sex == 1, "blue", "pink")
data$reason <- ifelse(data$status == 1, "검열", "사망")

# 시작나이/종료나이 설정
data_m <- data[ , c("time", "reason", "age", "color") ]
data_m$time <- data$age + data$time / 365

# 나이와 측정시간, 종료원인, 성별을 그린 그래프
ggplot( data_m, aes( x=age, y=seq_along(reason) ) ) +
  geom_segment(aes(xend=time, yend=seq_along(reason), color=color), linewidth=1) +
  geom_point(aes(x=time, shape=reason), size=2) +
  scale_color_identity() +
  labs(x = "나이", y = "측정자 번호") +
  theme_minimal()
