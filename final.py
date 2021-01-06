#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:22:13 2020

@author: figobaek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


pd.set_option("display.max_rows", 1000) 
pd.set_option("display.max_columns", 100)

df = pd.read_csv('/Users/figobaek/Desktop/project/project3/df1.csv')

df.isnull().any().sum()
df['성별(범주)'] = df['성별(범주)'].fillna('공동')

df.info()
df.iloc[:,0]
df = df.iloc[:,1:]
df.to_csv('/Users/figobaek/Desktop/project/project3/df1.csv')

df = pd.read_csv('/Users/figobaek/Desktop/project/project3/df1.csv')
df.info()

df


df['구매 계절'].value_counts().plot(kind='bar')

df[df['구매 계절'] == '가을']['완독여부'].value_counts()
df[df['구매 계절'] == '겨울']['완독여부'].value_counts()
df[df['구매 계절'] == '여름']['완독여부'].value_counts()
df[df['구매 계절'] == '봄']['완독여부'].value_counts()





# 쫙 펼처서 전처리(컬럼명, 범주화, 원핫)

df.info()
df['완독여부'].value_counts()
df['comp'] = df['완독여부'].map({'yes':1, 'no':0})
df['btype'] = df['책 형태(전자 or 종이)'].map({'p':1, 'e':0})

df.iloc[:,0].value_counts()
df.iloc[:,1].value_counts() # 작가명, 원핫

[i for i in agen.columns]

df.iloc[:,2].value_counts()
agen = pd.get_dummies(df['성별(범주)'], prefix='gen')
df = pd.concat([df, agen], axis=1)

df.iloc[:,3].value_counts() # 출판사, 원핫
df.iloc[:,4].value_counts() # 출판년도
df.iloc[:,5].value_counts()
df.iloc[:,6].value_counts() # 가격대 범주형으로 변경

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
bpr = pd.get_dummies(df['price'], prefix='pr')
df = pd.concat([df, bpr], axis=1)    
df.info()
    
df.iloc[:,7].value_counts() # map으로 완료 
df['btype'] = df['책 형태(전자 or 종이)'].map({'p':1, 'e':0})

df.iloc[:,8].value_counts() 
df.iloc[:,9].value_counts() 
df['score'] = round(df.iloc[:,9]) # 평점은 거의 높아서 의미 없어 보인다. 
df['score']

df.iloc[:,10].value_counts() # 표준화 정규화로!

df['oreview100'] = df.iloc[:,10] # 이상하게 복제되서 원본 새로운 컬럼에 담기

# 넘파이로 표준화 정규화 도전!
df['review100'] = (df.iloc[:,10] - np.mean(df.iloc[:,10],axis = 0))/ np.std(df.iloc[:,10],axis = 0)
print(std_data)

df.iloc[:,11].value_counts() #페이지수와 무게는 그냥 넣어보자. 궁금.
df.iloc[:,12].value_counts() 
df.iloc[:,13].value_counts() # 더미로!
genre = pd.get_dummies(df.iloc[:,13], prefix='genre')
df = pd.concat([df, genre], axis=1)

df.iloc[:,14].value_counts() # 년도 의미 없어 보임
df.iloc[:,15].value_counts() # 월,그냥 넣기
df.iloc[:,16].value_counts() # 더미로

season = pd.get_dummies(df.iloc[:,16], prefix='season')
df = pd.concat([df, season], axis=1)

df.info()    
    

a = []
for i in df['구매경로(범주)']:
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

prou = pd.get_dummies(df['p-route'], prefix='route')    
df = pd.concat([df, prou], axis=1)    
df.info()    
df.head()    
    
약식 전처리 완료

2. 디트에 넣을 컬럼 선정

df.columns    
df['comp']
df['btype']


df.iloc[:,-1]

# 숫자형 컬럼만 넣을 수 있다! 주의!!
feature_cols = ['oreview100', '세일즈포인트', '출판월', '가격', '전체쪽수',
       '무게(g)', 'btype', 'score', 'gen_공동', 'gen_남', 'gen_여',
       'pr_0 - 10000w', 'pr_10000 - 20000w', 'pr_20000w >',
       'route_목적달성', 'route_미디어추천', 'route_사람추천', 'route_스스로', 'route_알라딘알고리즘',
       'route_증정받음', 'season_가을', 'season_겨울', 'season_봄', 'season_여름']
    
x = df[feature_cols]
y = df['comp']    

x.info()
x.head()

# r 보내는 용도로 컬럼 정제한 데이터셋 합쳐 저장
dforr = pd.concat([x, y], axis=1)
dforr.to_csv('/Users/figobaek/Desktop/project/project3/dforr.csv')

# 여기부터 바로 시작해도 좋다. 
df = pd.read_csv('/Users/figobaek/Desktop/project/project3/dforr.csv')
df = df.iloc[:,1:]
df



# 숫자형 컬럼만 넣을 수 있다! 주의!! 트레인, 테스트 셋에 넣을 컬럼 지정

feature_cols = ['oreview100', '세일즈포인트', '출판월', '가격', '전체쪽수',
       '무게(g)', 'btype', 'score', 'gen_공동', 'gen_남', 'gen_여',
       'pr_0 - 10000w', 'pr_10000 - 20000w', 'pr_20000w >', 'route_목적달성', 
       'route_미디어추천', 'route_사람추천', 'route_스스로', 'route_알라딘알고리즘',
       'route_증정받음', 'season_가을', 'season_겨울', 'season_봄', 'season_여름']

x = df[feature_cols]
y = df['comp']    

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
                class_names=['못읽음', '다읽음'],
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



# 표준화 정규화를 한 차례 하고 다시 돌리니 테스트 정확도 상승 확인! 70까지!
# 확인된 중요 컬럼 - 무게, 출판월(?), 전체 쪽수(적어야 완), 가격(특정 가격대를 잘 읽네..),평점, 세일즈포인트
 
# 표준화 정규화 전 / 하나 / 다수 일때 비교하는 표 정리. 
# yes24 크롤링 까지 해서 테스트 하는 것으로 마무리. 











=====
회귀나무 해보자
=====

import numpy as np
from sklearn.tree import DecisionTreeRegressor    # 회귀나무 함수
import matplotlib.pyplot as plt

regr1 = DecisionTreeRegressor(max_depth = 2)
regr2 = DecisionTreeRegressor(max_depth = 5)

# 두 가지 회귀나무 적합
regr1.fit(x_train,y_train)
regr2.fit(x_train,y_train)

y_1 = regr1.predict(x_test)
y_2 = regr2.predict(x_test)


plt.figure()
plt.scatter(x_train, x_train, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(x_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(x_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()                        # sin함수의 예측을 목표로한다
y[::5] += 3 * (0.5 - rng.rand(16)) 

len(X)
len(x_train)
len(y)
len(y_train)

회귀나무 실패! 이건 끝나고 다시 공부하기.
========================









    
    
    
    
    
    
