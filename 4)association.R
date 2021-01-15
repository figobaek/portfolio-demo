
# �ѱ۱��� ���� 
par(family='AppleMyungjo')

# tidyverse ��ġ
install.packages('tidyverse')
library(tidyverse)

# stringr ��ġ 
install.packages('stringr')
library(stringr)

# konlp �ڿ��� ó�� ��Ű�� ������� �ڹ� ��ġ 
install.packages("rJava")
library(rJava)
install_jdk()
# ��� ����!! 
Sys.getenv("JAVA_HOME")
Sys.setenv(JAVA_HOME = '/Library/Java/JavaVirtualMachines/jdk-11.0.1.jdk/Contents/Home')

# �ڿ��� ó�� ��Ű�� �ҷ�����. 
install.packages('koNLP')
library(KoNLP) 
useSejongDic()

install.packages("multilinguer")
library(multilinguer)

install.packages(c('stringr', 'hash', 'tau', 'Sejong', 'RSQLite', 'devtools'), type = "binary")
install.packages("remotes")
remotes::install_github('haven-jeon/KoNLP', upgrade = "never", INSTALL_opts=c("--no-multiarch"))

# ������ �м� ���� �޼ҵ��
library(arules)
install.packages('arulesViz')
library(arulesViz)
install.packages('igraph')
library(igraph)

df32 <- read.csv("/Users/figobaek/Desktop/objective/portfolio/dfytcomments.csv", header = T, stringsAsFactors = F)
  
1) ������ �ɰ��� 
min <- df32[df32$pub == '������', 'comment']
md <- df32[df32$pub == '���е���', 'comment']
cb <- df32[df32$pub == 'â��', 'comment']



2) �ܾ� ���� ���� 
#�� �� ������ ��������� ��ó�� �߰�!
wlist <- list()
for(i in 1:length(min)){
  words <- SimplePos09(min[i])
  extracted <- str_match(words, '([��-???]+)/[NPM]')
  keyword <- extracted[,2]
  nouns <- keyword[!is.na(keyword)]
  nouns <- str_replace_all(nouns, '�λ�[��-???]+', '�λ����')
  nouns <- str_replace_all(nouns, '������Ƽ��+', 'mintv')
  nouns <- str_replace_all(nouns, '������[��-???]+', '������')
  nouns <- str_replace_all(nouns, 'mintv', '������tv')
  nouns <- str_replace_all(nouns, '[��-???]?���ͺ�[��-???]+', '���ͺ�')
  nouns <- str_replace_all(nouns, '[��-???]?�����л�[��-???]+', '�����л���')
  nouns <- str_replace_all(nouns, 'å��������', 'bookd')
  nouns <- str_replace_all(nouns, '[��-???]?������[��-???]+', '������')
  nouns <- str_replace_all(nouns, 'bookd', 'å������')
  nouns <- str_replace_all(nouns, '�б�������[��-???]+', '�б�������')
  nouns <- str_replace_all(nouns, '����', '�ʴ�')
  nouns <- str_replace_all(nouns, '��[��-???]+', '����')
  nouns <- str_replace_all(nouns, '�ιٲ����', '')
  nouns <- str_replace_all(nouns, '���ƿ�', '')
  nouns <- str_replace_all(nouns, '�����̳�[��-???]', '�����̳�')
  nouns <- str_replace_all(nouns, 'õ��[��-???]+', 'õ������̳�')
  nouns <- unique(nouns)
  nouns <- Filter(function(x){nchar(x)>1},nouns)
  nouns <- list(nouns)
  wlist <- c(wlist, nouns)
}
mintest <- unique(wlist) 
minl <- mintest

#�߰� ��ó�� 
#�˻� -> ��ü -> �˻� -> ��ü -> �� �˻�� ������ �ٸ� �ǹ� �з� 
#-> ��ü -> �з��ߴ� �� �ٽ� ��ü

str_extract_all(minl, '�����̳�[��-???]', simplify=T)
minl <- str_replace_all(minl, '�����̳�[��-???]', '�����̳�')

str_extract_all(mintest, 'õ��[��-???]+', simplify=T)
nouns <- str_replace_all(nouns, 'õ��[��-???]+', 'õ������̳�')

minl <- str_replace_all(minl, '�λ�[��-???]+', '�λ����')
str_extract_all(minl, '������[��-???]+', simplify=T)
str_replace_all(minl, '������[��-???]+', '������')
minl <- str_replace_all(minl, '������Ƽ��+', 'mintv')
minl <- str_replace_all(minl, '������[��-???]+', '������')
minl <- str_replace_all(minl, 'mintv', '������tv')
str_extract_all(minl, '������[��-???]+', simplify=T)
str_extract_all(minl, '[��-???]?���ͺ�[��-???]+', simplify=T)
minl <- str_replace_all(minl, '[��-???]?���ͺ�[��-???]+', '���ͺ�')
str_extract_all(minl, '[��-???]?�����л�[��-???]+', simplify=T)
minl <- str_replace_all(minl, '[��-???]?�����л�[��-???]+', '�����л���')
str_extract_all(minl, '[��-???]?������[��-???]+', simplify=T)
minl <- str_replace_all(minl, '[��-???]?�����л�[��-???]+', '�����л���')
minl <- str_replace_all(minl, 'å��������', 'bookd')
minl <- str_replace_all(minl, '[��-???]?������[��-???]+', '������')
minl <- str_replace_all(minl, 'bookd', 'å������')

str_extract_all(minl, '[��-???]?������[��-???]+', simplify=T)
minl <- str_replace_all(minl, '�б�������[��-???]+', '�б�������')
minl <- str_replace_all(minl, '���б�������', '�б�������')
# ������ �߰� �۾� �Ϸ�

# ���е���, â��� 
wlist <- list()
for(i in 1:length(md)){
  words <- SimplePos09(md[i])
  extracted <- str_match(words, '([��-???]+)/[NPM]')
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
  extracted <- str_match(words, '([��-???]+)/[NPM]')
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

3) apriori �޼ҵ忡 �´� Ʈ������ ���·� �������ֱ� 
# Ʈ������ ���·� ����
minwt <- as(minl, 'transactions') 
# ����ؼ� ���� 
summary(minwt) 
# ���̺� ����, arules �־�� ���ư�. 
minta <- crossTable(minwt) # ������
mdwt <- as(mdl, 'transactions') # ���е���
summary(mdwt) 
mdta <- crossTable(mdwt)
cbwt <- as(cbl, 'transactions') # â��
summary(cbwt) 
cbta <- crossTable(cbwt)

�󵵼� üũ �ð�ȭ # ���̺� ���� �� ���Ϸ� ����!
itemFrequencyPlot(minwt, support=0.1)
itemFrequencyPlot(minwt, topN=10)
itemFrequencyPlot(mdwt, support=0.05)
itemFrequencyPlot(mdwt, topN=10)
itemFrequencyPlot(cbwt, support=0.1)
itemFrequencyPlot(cbwt, topN=10)

4) ������ �м� �޼ҵ� ���� # Ʈ������ ���¸� �ִ� ���̴�.
minap <- apriori(minwt, parameter=list(supp=0.04, conf=0.04))
inspect(minap)
plot(minap,method="graph")
mdap <- apriori(mdwt, parameter=list(supp=0.04, conf=0.04))
inspect(mdap)
plot(mdap,method="graph")
cbap <- apriori(cbwt, parameter=list(supp=0.09, conf=0.08))
inspect(cbap)
plot(cbap,method="graph")


5) ������ �ð�ȭ
<������>
rules <- labels(minap, ruleSep=" ")
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rulemat <- do.call('rbind', rules)
ruleg <- graph.edgelist(rulemat, directed =F)

# dev.off(), �׷��� �� �׷��� ��
plot.igraph(ruleg, vertex.label=V(ruleg)$name, vertex.label.cex=1.0,
            vertex.label.color='black', vertex.size=20, vertex.color='chartreuse2',
            vertex.frame.color='chartreuse2')
<���е���>
rules <- labels(mdap, ruleSep=" ")
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rulemat <- do.call('rbind', rules)
ruleg <- graph.edgelist(rulemat, directed =F)

# dev.off(), �׷��� �� �׷��� ��
plot.igraph(ruleg, vertex.label=V(ruleg)$name, vertex.label.cex=1.0,
            vertex.label.color='black', vertex.size=20, vertex.color='orange',
            vertex.frame.color='orange')
<â��>
rules <- labels(cbap, ruleSep=" ")
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rulemat <- do.call('rbind', rules)
ruleg <- graph.edgelist(rulemat, directed =F)

# dev.off(), �׷��� �� �׷��� ��
plot.igraph(ruleg, vertex.label=V(ruleg)$name, vertex.label.cex=1.0,
            vertex.label.color='blue', vertex.size=20, vertex.color='violet',
            vertex.frame.color='black')




