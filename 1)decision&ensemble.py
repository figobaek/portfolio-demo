#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:22:13 2020

@author: figobaek
"""


# 라이브러리 호출, 한글 깨짐현상 방지 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 호출 행렬 최대치 조정 
pd.set_option("display.max_rows", 1000) 
pd.set_option("display.max_columns", 100)

# 수집해놓은 데이터 호출 (210111 세포, 평점, 리뷰갯수 표-정 돼있음)
## 성별 범주화는 저장을 해도 풀린다...다시 해줘야 함. 
df = pd.read_csv('/Users/figobaek/Desktop/objective/portfolio/df0112self.csv')

# 구조 및 값 파악
df.head()
df.info()


# 널 값 확인
df.isnull().any().sum()


# -- 데이터 전처리 -- 컬럼 범주화가 의미가 없을 수 있다. 머신러닝 넣기 전 숫자형으로 바꿔줘야 함. map 또는 dummy 메소드로 수치화 하자.


# 책 제목 글자수로 값 변경

b = []
for i in df['title']:
    a = i.split(' ')
    a = ''.join(a)
    b.append(len(a))
b

# 제목 글자수 컬럼 추가 
df['lentl'] = b


# 완독여부, 책 형태 원 핫 인코딩
df['comp'] = df['clear'].map({'yes':0, 'no':1})
df['btype'] = df['btype'].map({'p':1, 'e':0})


# 성별 원 핫 인코딩, 칼럼을 나눠 0 또는 1로 이 방법 또는 컬럼 범주화, 둘 중 하나로
# agen = pd.get_dummies(df['sex'], prefix='gen') 
# [i for i in agen.columns]
# df = pd.concat([df, agen], axis=1)

# 타입 바꿔주는 것으로 더 간편하게, 그러나 수치형으로.

df['sex'] = df['sex'].astype('category') 
df['sex'] = df['sex'].map({'공동':0, '남':1, '여':2})


# 가격 컬럼 범위 지정해 범주형으로 바꿔주기, 그리고 더미 메소드로 0 또는 1 값 컬럼으로 펼쳐주기 
df.iloc[:,6].value_counts() 

a = []
for i in df.iloc[:,6]:
    if i >= 0 and i <= 10000:
        a.append('0 - 10000w')
    elif i > 10000 and i <= 20000:
        a.append('10000 - 20000w')
    else:
        a.append('20000w >')
     
df['price'] = a
df['price'].value_counts()
# df['price'] = df['price'].astype('category') # 문자형이 되어서 범주화 의미 없다. 더미로 바꿔주기. 

# 더미화 
prange = pd.get_dummies(df.iloc[:,6], prefix='prange')
df = pd.concat([df, prange], axis=1)


# 세일즈포인트, 평점, 리뷰갯수는 모두 표준화 - 정규화 과정으로 전처리 

# 표준화 - 이어서 정규화까지 처리 (제목글자수, 쪽수, 무게 추가하기)
[표준화] 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

d8s = scale(df.iloc[:,8])
d9s = scale(df.iloc[:,9])
d10s = scale(df.iloc[:,10])
d11s = scale(df.iloc[:,11])
d12s = scale(df.iloc[:,12])
df['lentl'] = scale(df['lentl'])



[정규화]
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

# 원본 값은 두고 새로운 컬럼에 넣는다. 21~23번째 컬럼에 자리했다. 
df['spsn'] = minmax_scale(d8s) 
df['psn'] = minmax_scale(d9s)
df['rsn'] = minmax_scale(d10s)
df['pasn'] = minmax_scale(d11s)
df['wesn'] = minmax_scale(d12s)
df['lentl'] = minmax_scale(df['lentl'])

df.head()
df.info()
df.iloc[:,8:11]

#구매 계절 범주화 하기 
df['buys'] = df['buys'].astype('category')
df['buys'] = df['buys'].map({'봄':0, '여름':1, '가을':2, '겨울':3})


# 구매경로 범위 좁혀주기 

df['buyr'].value_counts()

a = []
for i in df['buyr']:
    if i in ('유튜브', '컨퍼런스','블로그','민음사tv','겨울서점','팟캐스트','릿터'):
        a.append('미디어추천')
    elif i in ('독서모임', '책모임', '취업준비'):
        a.append('목적달성')
    elif i in ('친구추천', '전문가추천'):
        a.append('사람추천')
    else:
        a.append(i)

df['p-route'] = a
df['p-route'].value_counts()    

# 6개의 범주로 분류됨. 그러나 스스로, 미디어 추천이 압도적으로 많아 유의미하지 않을 것으로 판단. 반씩 나누어 내적, 외적 요인으로 한 번 더 범주화 

b = []
for i in df['p-route']:
    if i in ('스스로', '목적달성'):
        b.append('내적요인')
    else:
        b.append('외적요인')

b
df['p-route'] = b
df['p-route'].value_counts()    

prou = pd.get_dummies(df['p-route'], prefix='route')    
df = pd.concat([df, prou], axis=1)    
    
# 수치형 변경, map 메소드 사용

df['p-route'] = df['p-route'].map({'내적요인':0, '외적요인':1})


# 장르를 넣기 위해 큰 범위로 재분류, 장르 수를 줄인다. 
df['genre'].value_counts().keys()

c = []
for i in df['genre']:
    if i in ('한국소설', '한국에세이', '영국문학', '독일문학', '교양인문학', '교양철학', '일본문학', '책읽기', '인문/사회', '동유럽문학','독서에세이'
             ,'여성학이론', '행복론', '심플라이프', '중남미문학', '외국과학소설', '프랑스문학', '에세이', '인문에세이', '한국사회비평', '인문학', '자기계발'
             ,'한국근현대사', '북유럽사', '영미문학', '서양철학', '한국과학소설', '영어학습법', '불교철학', '과학소설', '사회민주주의', '국제사회비평', '언론비평'):
        c.append('인문학')
    elif i in ('정치학일반', '환경문제', '사회학일반', '성공학', '경제전망', '경제비평', '세계경제전망', '정치학 일반', '사회복지', '노동문제', '외교정책'
               ,'정치인', '언론학', '광고/마케팅', '경제이야기', '성공', '법조인이야기', '주식펀드', '국제질서'):
        c.append('사회과학')
    else: 
        c.append('컴퓨터/모바일/수학')

df['genre2'] = c
wide = pd.get_dummies(df['genre2'], prefix='wide')
df = pd.concat([df, wide], axis=1)


# 재사용 위한 저장 
df.to_csv('/Users/figobaek/Desktop/objective/portfolio/df0112.csv', index=False)


#피처 넣을 후보 꼽아보기

제목
저자
성별
출판사
출판년도
출판월
가격
형태
세일즈포인트
평점
리뷰갯수+100자평
쪽수
무게
장르
구매년도
구매월
구매계절
구매경로
완독여부

[전처리한 컬림]
완독여부: map (y값을 사용)
책 형태 : map
세일즈포인트 : 표정
평점 : 표정
리뷰100자평 갯수 : 표정 
제목글자수 : 표정
구매경로 : map
sex : map
price : 더미 
쪽수 : 표정
무게 : 표정
구매계절 : map 
장르 : 범주화 -> 더미

[추가할 컬럼]


총 18개 컬럼으로 테스트. 

df.head()
df.info()
df.columns

2. 디트에 넣을 컬럼 선정

df.columns    

# 숫자형 컬럼만 넣을 수 있다! 주의!!
feature_cols = ['sex', 'buys', 'btype', 'spsn', 'psn', 'rsn',
       'lentl', 'p-route', 'pasn', 'wesn', 'prange_0 - 10000w',
       'prange_10000 - 20000w', 'prange_20000w >', 'wide_사회과학',
       'wide_인문학', 'wide_컴퓨터/모바일/수학']

feature 설명
sex : 성별 (0 - 공동저자, 1 - 남성, 2 - 여성)
buys : 구매계절 (0 - 봄, 1 - 여름, 2 - 가을, 3 - 겨울)
btype : 책 형태 (0 - 전자책, 1 - 종이책)
spsn : 세일즈포인트 표준화, 정규화
psn : 책 평점 표준화, 정규화
rsn : 리뷰 및 100자평 갯수 표준화, 정규화
*세일즈포인트, 평점, 리뷰 및 100자평 갯수는 알라딘 사이트 기준
lentl : 제목 글자수 표준화, 정규화
p-route : 구매경로(구매를 결정한 계기, 0 - 내적요인, 1 -외적요인)
pasn : 페이지 수 표준화, 정규화
wesn : 책 무게 표준화, 정규화
prange_0 - 10000w : 1만원 이하의 가격대
prange_10000 - 20000w : 1만원 이상, 2만원 이하 가격대
prange_20000w > : 2만원 이상 가격대
wide_사회과학, wide_인문학, wide_컴퓨터/모바일/수학 : 장르

comp : 완독여부 (0 - no, 1 - yes)

x = df[feature_cols]
y = df['comp']    

df['comp'].value_counts()
x.info()
x.head()


# 트레인, 테스트 셋 분리
from sklearn.model_selection import train_test_split
from collections import Counter
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

len(x_train)
len(x_test)
y_train
y_test
Counter(x_train)
Counter(y_train)
Counter(x_test)
Counter(y_test)

# 디시전 트리 모델링, 엔트로피 또는 지니계수, 얼마나 깊게 가지를 칠 것인가?
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy', max_depth=10)
model = DecisionTreeClassifier(criterion = 'gini', max_depth=10)
model.fit(x_train, y_train)

pred = model.predict(x_test)
model.score(x_train, y_train) 
model.score(x_test, y_test)

# 결과 확인, 혼동 행렬
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
model.classes_
confusion_matrix(y_train, model.predict(x_train)) # 모델링 데이터 그대로 예측
confusion_matrix(y_test, model.predict(x_test)) # 테스트 데이터 예측

# 중요 컬럼 시각화
len(model.feature_importances_)
len(feature_cols)
bpdf = pd.DataFrame({'항목':feature_cols,
              '중요도':model.feature_importances_})
dfs = bpdf.sort_values(by=['중요도'], ascending=False, axis=0)
dfs2 = dfs[dfs['중요도'] > 0]
dfs2.plot.barh(x='항목', y='중요도', color='green')

# 디시전트리 시각화

import pydotplus
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import Image

dot_data = export_graphviz(model, out_file=None,
                feature_names=feature_cols,
                class_names=['다읽음', '못읽음'],
                filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# 앙상블(배깅, 랜덤포레스트, 부스팅)
===
bagging & random forest
===
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 중요도 피처 뽑아낼 때 엔트로피냐, 인포메이션 게인이냐? / 트리 몇 개 만들거냐? / 한 번도 뽑히지 않은 애 모델 만들때 넣어? 
rm = RandomForestClassifier(criterion='entropy', n_estimators=100, oob_score=True)
rm = RandomForestClassifier(criterion='gini', n_estimators=100, oob_score=True)
rm = RandomForestClassifier(criterion='entropy', n_estimators=200, oob_score=True)
rm = RandomForestClassifier(criterion='entropy', n_estimators=100)
rm.fit(x_train, y_train)
accuracy_score(y_train, model.predict(x_train))
rm.score(x_train, y_train) # 트레인 100%
rm.score(x_test, y_test) # 테스트 55%

# 혼동행렬
confusion_matrix(y_train, rm.predict(x_train))
confusion_matrix(y_test, rm.predict(x_test))

# 분류 리포트 
print(classification_report(y_train, rm.predict(x_train))) # 100%
print(classification_report(y_test, rm.predict(x_test)))

===
boosting
===
from sklearn.ensemble import AdaBoostClassifier

am = AdaBoostClassifier(n_estimators=100)
am.fit(x_train, y_train)
accuracy_score(y_train, am.predict(x_train))
accuracy_score(y_test, am.predict(x_test))
confusion_matrix(y_test, am.predict(x_test))
print(classification_report(y_test, am.predict(x_test))) 

디시전 트리를 ada 부스터에 접목 
tm = DecisionTreeClassifier(criterion = 'entropy', max_depth=30)
tm = DecisionTreeClassifier(criterion = 'gini', max_depth=20)
ada_model = AdaBoostClassifier(base_estimator=tm, n_estimators=300)
ada_model.fit(x_train, y_train)
accuracy_score(y_train, ada_model.predict(x_train))
accuracy_score(y_test, ada_model.predict(x_test))
confusion_matrix(y_test, ada_model.predict(x_test))
print(classification_report(y_test, ada_model.predict(x_test))) 














    
    
    
    
    
    
