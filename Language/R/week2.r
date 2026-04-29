# 패키지 설치
install.packages("ggplot2")
install.packages("dplyr")

# 패키지 사용
library(ggplot2)
library(dplyr)

# 스터디 제공 데이터 불러오기
data <- read.csv("data_modified.csv")

# 각 대륙별 수치 총합 구하고 열 이름 재지정
contsum <- aggregate(data$Value, by=list(data$ParentLocation, data$Period), FUN=sum)
colnames(contsum) <- c("continents", "years", "values")

# 비율 구하기
contratio <- contsum %>% group_by(years) %>% mutate( ratio = values / sum(values) )

# 연도별/대륙별 수치 비율 그래프, 막대그래프 형식에 항목별 이름 추가, 전체를 100%로 하는 스케일
ggplot( contratio, aes(x = years, y = ratio, fill = continents) ) +
    geom_bar(stat = "identity") +
    labs(x = "연도", y = "비율", fill = "대륙") +
    scale_y_continuous( labels = scales::percent_format(scale = 100) )
