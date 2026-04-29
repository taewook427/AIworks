# 패키지 설치
install.packages("ggplot2")

# 패키지 사용
library(ggplot2)

# 데이터 가져오고 유효한 담배, ICU 데이터만 거르기
data <- read.csv("covid_data.csv")
data <- data[ , c("TOBACCO", "ICU") ]
data <- data[data$TOBACCO == 1 | data$TOBACCO == 2,]
data <- data[data$ICU == 1 | data$ICU == 2,]

# 담배X ICUX, 담배X ICUO, 담배O ICUX, 담배O ICUO
data <- c(
nrow(data[data$TOBACCO == 2 & data$ICU == 2,]),
nrow(data[data$TOBACCO == 2 & data$ICU == 1,]),
nrow(data[data$TOBACCO == 1 & data$ICU == 2,]),
nrow(data[data$TOBACCO == 1 & data$ICU == 1,])
)

# 그릴 데이터프레임 생성
todraw <- data.frame(type=c("비흡연 non-ICU", "비흡연 ICU", "흡연 non-ICU", "흡연 ICU"), data=data)

# 로그스케일로 본 각 상황, ICU 비율이 거의 10:1로 동일하다
ggplot(todraw, aes(x=type, y=data)) +
  geom_bar(stat="identity", fill="steelblue") +
  scale_y_log10() +
  theme_minimal() +
  labs(x="Type", y="Number", title="Smoking & ICU Bar Chart")

# 흡연과 ICU 여부가 독립적인지 검정 (연관 if p < 0.05)
totest <- matrix(data, nrow=2)
print(chisq.test(totest))
