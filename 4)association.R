
# 한글깨짐 방지 
par(family='AppleMyungjo')

# tidyverse 설치
install.packages('tidyverse')
library(tidyverse)

# stringr 설치 
install.packages('stringr')
library(stringr)

# konlp 자연어 처리 패키지 사용위한 자바 설치 
install.packages("rJava")
library(rJava)
install_jdk()
# 경로 설정!! 
Sys.getenv("JAVA_HOME")
Sys.setenv(JAVA_HOME = '/Library/Java/JavaVirtualMachines/jdk-11.0.1.jdk/Contents/Home')

# 자연어 처리 패키지 불러오기. 
install.packages('koNLP')
library(KoNLP) 
useSejongDic()

install.packages("multilinguer")
library(multilinguer)

install.packages(c('stringr', 'hash', 'tau', 'Sejong', 'RSQLite', 'devtools'), type = "binary")
install.packages("remotes")
remotes::install_github('haven-jeon/KoNLP', upgrade = "never", INSTALL_opts=c("--no-multiarch"))

# 연관성 분석 관련 메소드들
library(arules)
install.packages('arulesViz')
library(arulesViz)
install.packages('igraph')
library(igraph)

df32 <- read.csv("/Users/figobaek/Desktop/objective/portfolio/dfytcomments.csv", header = T, stringsAsFactors = F)
  
1) 데이터 쪼개기 
min <- df32[df32$pub == '민음사', 'comment']
md <- df32[df32$pub == '문학동네', 'comment']
cb <- df32[df32$pub == '창비', 'comment']



2) 단어 조각 내기 
#세 번 돌리기 민음사용은 전처리 추가!
wlist <- list()
for(i in 1:length(min)){
  words <- SimplePos09(min[i])
  extracted <- str_match(words, '([가-???]+)/[NPM]')
  keyword <- extracted[,2]
  nouns <- keyword[!is.na(keyword)]
  nouns <- str_replace_all(nouns, '인상[가-???]+', '인상깊다')
  nouns <- str_replace_all(nouns, '민음사티비를+', 'mintv')
  nouns <- str_replace_all(nouns, '민음사[가-???]+', '민음사')
  nouns <- str_replace_all(nouns, 'mintv', '민음사tv')
  nouns <- str_replace_all(nouns, '[가-???]?인터뷰[가-???]+', '인터뷰')
  nouns <- str_replace_all(nouns, '[가-???]?고등학생[가-???]+', '고등학생들')
  nouns <- str_replace_all(nouns, '책디자인이', 'bookd')
  nouns <- str_replace_all(nouns, '[가-???]?디자인[가-???]+', '디자인')
  nouns <- str_replace_all(nouns, 'bookd', '책디자인')
  nouns <- str_replace_all(nouns, '학교도서관[가-???]+', '학교도서관')
  nouns <- str_replace_all(nouns, '느리', '늦다')
  nouns <- str_replace_all(nouns, '느[가-???]+', '느낌')
  nouns <- str_replace_all(nouns, '로바꿔야할', '')
  nouns <- str_replace_all(nouns, '같아욥', '')
  nouns <- str_replace_all(nouns, '디자이너[가-???]', '디자이너')
  nouns <- str_replace_all(nouns, '천재[가-???]+', '천재디자이너')
  nouns <- unique(nouns)
  nouns <- Filter(function(x){nchar(x)>1},nouns)
  nouns <- list(nouns)
  wlist <- c(wlist, nouns)
}
mintest <- unique(wlist) 
minl <- mintest

#추가 전처리 
#검색 -> 교체 -> 검색 -> 교체 -> 한 검색어에 잡히는 다른 의미 분류 
#-> 교체 -> 분류했던 것 다시 교체

str_extract_all(minl, '디자이너[가-???]', simplify=T)
minl <- str_replace_all(minl, '디자이너[가-???]', '디자이너')

str_extract_all(mintest, '천재[가-???]+', simplify=T)
nouns <- str_replace_all(nouns, '천재[가-???]+', '천재디자이너')

minl <- str_replace_all(minl, '인상[가-???]+', '인상깊다')
str_extract_all(minl, '민음사[가-???]+', simplify=T)
str_replace_all(minl, '민음사[가-???]+', '민음사')
minl <- str_replace_all(minl, '민음사티비를+', 'mintv')
minl <- str_replace_all(minl, '민음사[가-???]+', '민음사')
minl <- str_replace_all(minl, 'mintv', '민음사tv')
str_extract_all(minl, '민음사[가-???]+', simplify=T)
str_extract_all(minl, '[가-???]?인터뷰[가-???]+', simplify=T)
minl <- str_replace_all(minl, '[가-???]?인터뷰[가-???]+', '인터뷰')
str_extract_all(minl, '[가-???]?고등학생[가-???]+', simplify=T)
minl <- str_replace_all(minl, '[가-???]?고등학생[가-???]+', '고등학생들')
str_extract_all(minl, '[가-???]?디자인[가-???]+', simplify=T)
minl <- str_replace_all(minl, '[가-???]?고등학생[가-???]+', '고등학생들')
minl <- str_replace_all(minl, '책디자인이', 'bookd')
minl <- str_replace_all(minl, '[가-???]?디자인[가-???]+', '디자인')
minl <- str_replace_all(minl, 'bookd', '책디자인')

str_extract_all(minl, '[가-???]?도서관[가-???]+', simplify=T)
minl <- str_replace_all(minl, '학교도서관[가-???]+', '학교도서관')
minl <- str_replace_all(minl, '학학교도서관', '학교도서관')
# 민음사 추가 작업 완료

# 문학동네, 창비용 
wlist <- list()
for(i in 1:length(md)){
  words <- SimplePos09(md[i])
  extracted <- str_match(words, '([가-???]+)/[NPM]')
  keyword <- extracted[,2]
  nouns <- keyword[!is.na(keyword)]
  nouns <- unique(nouns)
  nouns <- Filter(function(x){nchar(x)>1},nouns)
  nouns <- list(nouns)
  wlist <- c(wlist, nouns)
}

mdl <- unique(wlist)  
mdl

wlist <- list()
for(i in 1:length(cb)){
  words <- SimplePos09(cb[i])
  extracted <- str_match(words, '([가-???]+)/[NPM]')
  keyword <- extracted[,2]
  nouns <- keyword[!is.na(keyword)]
  nouns <- unique(nouns)
  nouns <- Filter(function(x){nchar(x)>1},nouns)
  nouns <- list(nouns)
  wlist <- c(wlist, nouns)
}

cbl <- unique(wlist)  
cbl

minl
mdl
cbl

=====================================

3) apriori 메소드에 맞는 트렌젝션 형태로 변경해주기 
# 트렌젝션 형태로 변경
minwt <- as(minl, 'transactions') 
# 요약해서 보기 
summary(minwt) 
# 테이블 생성, arules 있어야 돌아감. 
minta <- crossTable(minwt) # 민음사
mdwt <- as(mdl, 'transactions') # 문학동네
summary(mdwt) 
mdta <- crossTable(mdwt)
cbwt <- as(cbl, 'transactions') # 창비
summary(cbwt) 
cbta <- crossTable(cbwt)

빈도수 체크 시각화 # 테이블 생성 전 파일로 가능!
itemFrequencyPlot(minwt, support=0.1)
itemFrequencyPlot(minwt, topN=10)
itemFrequencyPlot(mdwt, support=0.05)
itemFrequencyPlot(mdwt, topN=10)
itemFrequencyPlot(cbwt, support=0.1)
itemFrequencyPlot(cbwt, topN=10)

4) 연관성 분석 메소드 적용 # 트랜젝션 형태를 넣는 것이다.
minap <- apriori(minwt, parameter=list(supp=0.04, conf=0.04))
inspect(minap)
plot(minap,method="graph")
mdap <- apriori(mdwt, parameter=list(supp=0.04, conf=0.04))
inspect(mdap)
plot(mdap,method="graph")
cbap <- apriori(cbwt, parameter=list(supp=0.09, conf=0.08))
inspect(cbap)
plot(cbap,method="graph")


5) 연관성 시각화
<민음사>
rules <- labels(minap, ruleSep=" ")
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rulemat <- do.call('rbind', rules)
ruleg <- graph.edgelist(rulemat, directed =F)

# dev.off(), 그래프 안 그려질 때
plot.igraph(ruleg, vertex.label=V(ruleg)$name, vertex.label.cex=1.0,
            vertex.label.color='black', vertex.size=20, vertex.color='chartreuse2',
            vertex.frame.color='chartreuse2')
<문학동네>
rules <- labels(mdap, ruleSep=" ")
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rulemat <- do.call('rbind', rules)
ruleg <- graph.edgelist(rulemat, directed =F)

# dev.off(), 그래프 안 그려질 때
plot.igraph(ruleg, vertex.label=V(ruleg)$name, vertex.label.cex=1.0,
            vertex.label.color='black', vertex.size=20, vertex.color='orange',
            vertex.frame.color='orange')
<창비>
rules <- labels(cbap, ruleSep=" ")
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rulemat <- do.call('rbind', rules)
ruleg <- graph.edgelist(rulemat, directed =F)

# dev.off(), 그래프 안 그려질 때
plot.igraph(ruleg, vertex.label=V(ruleg)$name, vertex.label.cex=1.0,
            vertex.label.color='blue', vertex.size=20, vertex.color='violet',
            vertex.frame.color='black')





