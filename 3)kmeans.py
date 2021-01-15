#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:03:00 2020

@author: figobaek
"""

# 메소드 호출
from selenium import webdriver
from selenium import webdriver as wd
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
from urllib import request; from urllib import parse
import pandas as pd

# 영상 별 크롤링
===
민음사
===
url = 'https://www.youtube.com/watch?v=TI8gOV7nHNk&t=18s'
driver = webdriver.Chrome('/usr/local/bin/chromedriver')
driver.implicitly_wait(1)
driver.get(url)
driver.implicitly_wait(3)
driver.execute_script("window.scrollTo(0, 20);")  

last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    
    time.sleep(2.0)  
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2.0)       # 인터발 1이상으로 줘야 데이터 취득가능(롤링시 데이터 로딩 시간 때문)                                               


    new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
minid = [i.text.strip() for i in soup.select('#author-text > span')]
mintx = [i.text.strip() for i in soup.select('#content-text')]

=====
문학동네
=====
url = 'https://www.youtube.com/watch?v=Oh4v0OW7YW8'
driver = webdriver.Chrome('/usr/local/bin/chromedriver')
driver.implicitly_wait(1)
driver.get(url)
driver.implicitly_wait(3)
driver.execute_script("window.scrollTo(0, 20);")  

last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(3.0)       
    new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
mdid = [i.text.strip() for i in soup.select('#author-text > span')]
mdtx = [i.text.strip() for i in soup.select('#content-text')]



==
창비
==
url = 'https://www.youtube.com/watch?v=1iRB4DUN-HU'
driver = webdriver.Chrome('/usr/local/bin/chromedriver')
driver.implicitly_wait(1)
driver.get(url)
driver.implicitly_wait(1)
driver.execute_script("window.scrollTo(0, 20);")  

#잘 안 돼서 와일문 전에 살짝 내려줌. 
last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight-50);")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(3.0)       
    new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
cbid = [i.text.strip() for i in soup.select('#author-text > span')]
cbtx = [i.text.strip() for i in soup.select('#content-text')]

===
데프화, 붙이기
===
mindf = pd.DataFrame({'id':minid, 'comment':mintx})
mindf['pub'] = '민음사'
mindf.head()
mindf.info()

mddf = pd.DataFrame({'id':mdid, 'comment':mdtx})
mddf['pub'] = '문학동네'
mddf.head()
mddf.info()

cbdf = pd.DataFrame({'id':cbid, 'comment':cbtx})
cbdf['pub'] = '창비'
cbdf.head()
cbdf.info()

pro32df = pd.concat([mindf, mddf, cbdf], axis=0)
pro32df.head()
pro32df.info()

pro32df.to_csv('/Users/figobaek/Desktop/project/project3/dfytcomments.csv', index=False)




=====
전처리 시작
====
p32 = pd.read_csv('/Users/figobaek/Desktop/objective/portfolio/dfytcomments.csv')

p32.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 421 entries, 0 to 420
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   id       421 non-null    object
 1   comment  421 non-null    object
 2   pub      421 non-null    object
dtypes: object(3)
memory usage: 10.0+ KB

p32.head()
               id                                            comment  pub
0       사부작대는비단털쥐  뭔가,... 내향형 김이나작사가님 같은 느낌이... 세상 초탈한듯한 분위기랑 맡은 ...  민음사
1  Ssari Flamingo  디자이너분 진짜 일 잘하시고 사회생활에 찌들어서 선을 잘 긋고 딱 부러진 느낌이 나...  민음사
2             AUF  민음사 책 좋아하는 이유 중 하나가 이 분께 있었네...그런데 그게 사심과 자기 확...  민음사
3  Cate Blanchett  천재디자이너님 너무 솜방망이 같으세요.. 무른 듯 물러 보이지만 맞으면 너무 아파요...  민음사
4         sy park             보다보니까 저작권팀에서도 한분 나오셔야하실것같아욥...궁금하네요...  민음사

# 결측치 체크 - 없음
p32.isnull().any()

import re
from konlpy.tag import Hannanum

# 한나눔 형태소 분석기 클래스 생성하기
hannanum = Hannanum()

# 한 줄씩 쪼개고 전처리 그리고 다시 통합 리스트로 정리 
docs = [ ]
for i in range(421):
     a = hannanum.nouns(p32['comment'][i])
     a = [re.sub('[^A-z가-힣]', "", i) for i in a] # ㅋ나 ㅎ도 없애준다.
     a = [i.strip() for i in a ]
     a = [i for i in a if len(i) > 1 and i != '' ]
     docs.append(a)
 
# 형태가 한 리스트 안에 분리된 단어 조각을 한 줄로 다시 붙여주기(원래 한 문장 이었으니까)
for i in range(len(docs)):
     docs[i] = ' '.join(docs[i])



------------------
# 명사를 벡터로 만들기

from sklearn.feature_extraction.text import CountVectorizer
news_vec = CountVectorizer()
news_x = news_vec.fit_transform(docs)
news_x
 
# 벡터를 데이터 프레임으로 만들기
news_df = pd.DataFrame(news_x.toarray(), columns = news_vec.get_feature_names())
news_df.info() # 컬럼 1508개였음
 -----------------
 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 주성분 분석하기  - 표준화 정규화처럼 수치를 변형하는 것. 군집 알고리즘 사용이 가능한 두 개의 컬럼으로 변경. 
news_pca = PCA(n_components = 2)
news_principalComponents = news_pca.fit_transform(news_df)

# 주성분 분석의 결과를 데이터 프레임으로 만들기 /  421개 행의 컬럼을 두 개로. 
news_df_pca = pd.DataFrame(data = news_principalComponents,columns = ["PC1", "PC2"])
news_df_pca.index = p32['pub']




# 여기서 잠시 엘보우로 적당한 군집 수 확인
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(news_df_pca) 
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("군집수")
plt.ylabel("SSE")
plt.show()

# 엘보우 막대 그래프화(실루엣)
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline
rc('font', family='AppleMyungjo')
plt.rcParams['axes.unicode_minus'] = False

silhouette_scores = [] 
for n_cluster in range(2, 7):
    silhouette_scores.append( 
        silhouette_score(news_df_pca, KMeans(n_clusters = n_cluster).fit_predict(news_df_pca))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6] 
plt.bar(k, silhouette_scores) 
plt.xlabel('군집수', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 


# 시각화
kmeans = KMeans(n_clusters=2)
kmeans.fit(news_df_pca)

plt.scatter(news_df_pca['PC1'], news_df_pca['PC2'],  
           c = KMeans(n_clusters = 2).fit_predict(news_df_pca), 
           cmap =plt.cm.winter) 
plt.show() 


# 위 데이터 그대로 진행해도 되지만 가설 검증을 위해 데이터 셋을 분리해줌.
minpca = news_df_pca[news_df_pca.index == '민음사']
mdpca = news_df_pca[news_df_pca.index == '문학동네']
cbpca = news_df_pca[news_df_pca.index == '창비']

minpca.iloc[:,0]
minpca.iloc[:,1]

mdpca.iloc[:,0]
mdpca.iloc[:,1]

cbpca.iloc[:,0]
cbpca.iloc[:,1]

# 민음사 
plt.scatter(minpca.iloc[:,0],minpca.iloc[:,1], s=20, 
            c="green", label="민음사")
# 문학동네
plt.scatter(mdpca.iloc[:,0], mdpca.iloc[:,1], s=20, 
            c= "orange", label="문학동네")
# 창비
plt.scatter(cbpca.iloc[:,0], cbpca.iloc[:,1], s=20, 
            c= "red",label = "창비")
plt.legend()
plt.show()


























