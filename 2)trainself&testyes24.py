#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:59:38 2020

@author: figobaek
"""

pd.set_option("display.max_rows", 1000) 
pd.set_option("display.max_columns", 100)

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


books

def yes24DataReader(CategoryNumber):

    root_url = 'http://www.yes24.com'

    url_1 = 'http://www.yes24.com/24/category/Bestseller?CategoryNumber='
    url_2 = '&sumgb=06'
    url_3 = '&PageNumber='
    url_set = url_1 + CategoryNumber + url_2 + url_3 

    book_list=[]

    for i in range(1,6):

        url = url_set + str(i)

        res = requests.post(url)
        soup = BeautifulSoup(res.text, 'html5lib')
        tag = '#category_layout > tbody > tr > td.goodsTxtInfo > p:nth-child(1) > a:nth-child(1)'
        books = soup.select(tag)

        # 수집 중인 페이지 번호 출력
        print('# Page', i)

        # 개별 도서 정보 수집
        for book in books:

            sub_url = root_url + book.attrs['href']
            sub_res = requests.post(sub_url)
            sub_soup = BeautifulSoup(sub_res.text, 'html5lib')

            tag_name = '#yDetailTopWrap > div.topColRgt > div.gd_infoTop > div > h2'
            
            tag_author = '#yDetailTopWrap > div.topColRgt > div.gd_infoTop > span.gd_pubArea > span.gd_auth > a'
            tag_author2 = '#yDetailTopWrap > div.topColRgt > div.gd_infoTop > span.gd_pubArea > span.gd_auth'
            
            tag_publisher = '#yDetailTopWrap > div.topColRgt > div.gd_infoTop > span.gd_pubArea > span.gd_pub > a'
            
            tag_date = '#yDetailTopWrap > div.topColRgt > div.gd_infoTop > span.gd_pubArea > span.gd_date'
            tag_point = '#infoset_reviewTop > div.review_starWrap > div.gd_rating > em.yes_b'
            
            tag_sales = '#yDetailTopWrap > div.topColRgt > div.gd_infoTop > span.gd_ratingArea > span.gd_sellNum'

            tag_price = '#yDetailTopWrap > div.topColRgt > div.gd_infoBot > div.gd_infoTbArea > div:nth-child(3) > table > tbody > tr:nth-child(2) > td > span > em'
            tag_price2 = '#yDetailTopWrap > div.topColRgt > div.gd_infoBot > div.gd_infoTbArea > div:nth-child(4) > table > tbody > tr:nth-child(2) > td > span > em'

            tag_review1 = '#yDetailTabNavWrap > div > div.gd_tabBar > ul > li:nth-child(2) > a > em.txt_num'

            tag_page = '#infoset_specific > div.infoSetCont_wrap > div > table > tbody > tr:nth-child(2) > td'
            tag_weight = '#infoset_specific > div.infoSetCont_wrap > div > table > tbody > tr:nth-child(2) > td'

            # 기본적인 예외처리를 통한 데이터 수집
            name = sub_soup.select(tag_name)[0].text

            try:
                author = sub_soup.select(tag_author)[0].text
            except:
                author = sub_soup.select(tag_author2)[0].text.strip('\n').strip().replace(' 저','')


            publisher = sub_soup.select(tag_publisher)[0].text
            date = sub_soup.select(tag_date)[0].text.replace('년 ','-').replace('월 ','-').replace('일','')
            
            try:
                point = sub_soup.select(tag_point)[0].text
                point = float(point)
            except:
                point = int(0)
            
            
            try:
                sales = ''.join(sub_soup.select(tag_sales)[0].text.split(','))
                if '판매지수' in sales:
                    sales = ''.join(sub_soup.select(tag_sales)[0].text.strip().strip('|').strip().lstrip('판매지수 ').rstrip(' 판매지수란?').split(','))
                    sales = int(sales)
                else :
                    sales = int(0)
            except:
                sales = int(0)
            


            try:
                price = sub_soup.select(tag_price)[0].text.replace(',','')
                price = int(price)
            except:
                try:
                    price = sub_soup.select(tag_price2)[0].text.replace(',','')
                    price = int(price)
                except:
                    price = int(0)
           
            
            try:
                review1 = sub_soup.select(tag_review1)[0].text
                review1 = review1.lstrip('(').rstrip(')').split('/')
                review1 = sum([int(i) for i in review1])
                review1 = int(review1)
            except:
                review1 = int(0)
            
            # 불러낸다음 뒤에 붙은 문자 떼어내고 변수 저장
            page = ''.join(sub_soup.select(tag_page)[0].text.split(','))
            if '쪽' in page:
                if '확인' in page:
                    page = None
                else :
                    page = page.split('|')[0].strip().replace('쪽','')
                    page = int(page)
            else :
                page = None
           

            weight = ''.join(sub_soup.select(tag_weight)[0].text.split(','))
            if 'g' in weight:
                weight = weight[:weight.find('g')].split('|')[1].strip()
                weight = int(weight)
            else :
                weight = int(0)
            
            
            book_list.append([name, author, publisher, date,
                              int(sales), int(price), int(page),
                              int(weight), int(point), int(review1) ])

            print('=========>', name)

    # 데이터프레임 컬럼명 지정
    colList = ['name',  'author', 'publisher', 'date',
               'sales', 'price', 'page',
               'weight', 'point', 'review1']

    # 데이터프레임으로 변환
    df = pd.DataFrame(np.array(book_list), columns=colList)

    return df


CategoryNum='001'
df = yes24DataReader(CategoryNum)


CategoryNum='001'


        # 월 별로 수집된 데이터를 CSV 형식 파일로 저장
        df.to_csv(str(year)+'_'+str(month)+'_'+str(CategoryNum)+'.csv', index=False, encoding='CP949')


# 베스트 샐러 국내 100선 수집

1. nan 값을 컬럼 중간값으로 대체

df

CategoryNum='001'
df = yes24DataReader(CategoryNum)


결국 수동으로 ㅜㅜ
df.page = df.page.astype(int)
df.sales = df.sales.astype(int)
df.price = df.price.astype(int)
df.weight = df.weight.astype(int)
df.point = df.point.astype(int)
df.review1 = df.review1.astype(int)

df.info()
df.head()
df.describe()
df.isnull().any()


# null 값을 0으로 대체해준 데이터가 있는 컬럼은 weight, point, review1
-->리뷰는 없는 게 맞기에 그대로 두고, 별점이 없는 건 해당 사이트에서 구매자들이 굳이 별점을 매기지 않은 것이므로 그대로 둠.
-->무게 정보 없는 것만 해당 데이터 셋 책들의 무게 평균 값으로 채우기. 

df = df.iloc[:,:-1]

a = []
for i in range(len(df['weight'])):
    if df['weight'][i] == 0:
        a.append(round(df['weight'].mean()))
    else:
        a.append(df['weight'][i])

df['weight2'] = a


df.to_csv('/Users/figobaek/Desktop/project/project3/df3.csv', index=False)
df3 = pd.read_csv('/Users/figobaek/Desktop/objective/itwill/project/project3/dfyes1.csv')
df3


df3.info()

최소 컬럼만 맞춰 돌려보기

1. 제목 글자수 표정
2. 세일즈 포인트 표정
3. 페이지수 표정
4. 책 무게 표정
5. 평점 표정
6. 리뷰갯수 표정 
7~9. 가격 더미 

10. 완독여부 랜덤 배정, 테스트용. 

9개 컬럼 만들고 하나는 결과.

df3.info()

# 제목 글자수! 아주 좋은 접근. 

b = []
for i in df3['name']:
    a = i.split(' ')
    a = ''.join(a)
    b.append(len(a))
b

# 제목 글자수 컬럼 추가 
df3['lentl'] = b

# 가격 컬럼 범위 지정해 범주형으로 바꿔주기, 그리고 더미 메소드로 0 또는 1 값 컬럼으로 펼쳐주기 
df3.iloc[:,6].value_counts() 

a = []
for i in df3.iloc[:,5]:
    if i >= 0 and i <= 10000:
        a.append('0 - 10000w')
    elif i > 10000 and i <= 20000:
        a.append('10000 - 20000w')
    else:
        a.append('20000w >')
     
df3['pr'] = a
df3['pr'].value_counts()
# df['price'] = df['price'].astype('category') # 문자형이 되어서 범주화 의미 없다. 더미로 바꿔주기. 

# 더미화 
prange = pd.get_dummies(df3.iloc[:,13], prefix='prange')
df3 = pd.concat([df3, prange], axis=1)
df3.info()


# 제목 글자수, 세일즈포인트, 페이지수, 무게, 평점, 리뷰갯수는 모두 표준화 - 정규화 

# 표준화 - 이어서 정규화까지 처리 
[표준화] 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

d4s = scale(df3.iloc[:,4])
d5s = scale(df3.iloc[:,5])
d6s = scale(df3.iloc[:,6])
d7s = scale(df3.iloc[:,7])
d8s = scale(df3.iloc[:,8])
d12s = scale(df3.iloc[:,12])

[정규화]
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

# 원본 값은 두고 새로운 컬럼에 넣는다. 21~23번째 컬럼에 자리했다. 

df3['spsn'] = minmax_scale(d4s) 
df3['psn'] = minmax_scale(d5s)
df3['pasn'] = minmax_scale(d6s)
df3['wesn'] = minmax_scale(d7s)
df3['rsn'] = minmax_scale(d8s)
df3['lentl'] = minmax_scale(d12s)

df3.head()
df3.info()
df3.iloc[:,8:11]

# 완독여부 랜덤 생성 (50:50)

df3['comp'] = pd.DataFrame(np.random.randint(0, 2, 100))
df3['comp'].value_counts()



# 여기부터 바로 시작해도 좋다. 
df3.to_csv('/Users/figobaek/Desktop/objective/portfolio/df0112yes.csv', index=False)
df3 = pd.read_csv('/Users/figobaek/Desktop/objective/portfolio/df0112yes.csv')
df3

df3.columns

# 넣을 컬럼 정하고 x, y 나눠주기
# 자체 데이터 결과 기준 상위 컬럼만 편집. 

feature 설명

spsn : 세일즈포인트 표준화, 정규화
psn : 책 평점 표준화, 정규화
rsn : 리뷰 및 100자평 갯수 표준화, 정규화
*세일즈포인트, 평점, 리뷰 및 100자평 갯수는 yes24 사이트 기준
lentl : 제목 글자수 표준화, 정규화
pasn : 페이지 수 표준화, 정규화
wesn : 책 무게 표준화, 정규화
prange_0 - 10000w : 1만원 이하의 가격대
prange_10000 - 20000w : 1만원 이상, 2만원 이하 가격대
prange_20000w > : 2만원 이상 가격대


df.info()

feature_cols = ['lentl', 'prange_0 - 10000w', 'prange_10000 - 20000w', 'prange_20000w >', 'spsn',
       'psn', 'pasn', 'wesn', 'rsn']
    
# 자체 데이터 트레이닝 후 스크랩한 데이터로 테스트 해보기 

x_train = df[feature_cols]
y_train = df['comp']  
x_test = df3[feature_cols]
y_test = df3['comp']


len(x_train)
len(x_test)
y_train
y_test

Counter(x_train)
Counter(y_train)
Counter(x_test)
Counter(y_test)

균등분배 잘 됨. 

from sklearn.tree import DecisionTreeClassifier

# 디시전 트리 모델링, 엔트로피로 분류
model = DecisionTreeClassifier(criterion = 'entropy', max_depth=20)
model = DecisionTreeClassifier(criterion = 'gini', max_depth=10)
model.fit(x_train, y_train)

껍데기를 만들어서 피팅 시키자. 

pred = model.predict(x_test)
pred

model.score(x_train, y_train) 
model.score(x_test, y_test)

맥스뎁스를 조금씩 올려가면서 정확도 어떻게 바뀌는지 체크하기 

from sklearn.metrics import confusion_matrix

# 결과 확인, 혼동 행렬
confusion_matrix(y_test, pred)
model.classes_

실제 모델을 넣고 예측해보면? 답과 예측모델에 넣어서 돌려본 것과 비교 
confusion_matrix(y_train, model.predict(x_train))
confusion_matrix(y_test, model.predict(x_test))

@@### 와우!! 컬럼 세분화(원핫, 표준화정규화) + 엔트로피 + 맥스뎁스 10 테스트 정확도 급상승!!

# 중요 컬럼 확인

len(model.feature_importances_)
len(feature_cols)
model.feature_importances_
feature_cols

bpdf = pd.DataFrame({'항목':feature_cols,
              '중요도':model.feature_importances_})

dfs = bpdf.sort_values(by=['중요도'], ascending=False, axis=0)
dfs2 = dfs[dfs['중요도'] > 0]

dfs2.plot.barh(x='항목', y='중요도', color='red')

# 중요도 : 책 무게 -> 가격 -> 리뷰수 -> 페이지수

무게가 참 중요하구나~ 

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



# 앙상블로 !


===
bagging & random forest
===

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

중요도 피처 뽑아낼 때 엔트로피냐, 인포메이션 게인이냐? / 트리 몇 개 만들거냐? / 한 번도 뽑히지 않은 애 모델 만들때 넣어? 
rm = RandomForestClassifier(criterion='entropy', n_estimators=100, oob_score=True)
rm = RandomForestClassifier(criterion='gini', n_estimators=100, oob_score=True)
rm = RandomForestClassifier(criterion='entropy', n_estimators=200, oob_score=True)
rm = RandomForestClassifier(criterion='entropy', n_estimators=100)

rm.fit(x_train, y_train)
accuracy_score(y_train, model.predict(x_train))
rm.score(x_train, y_train) # 트레인 100%
rm.score(x_test, y_test) # 테스트 55%

-혼동행렬
confusion_matrix(y_train, rm.predict(x_train))
confusion_matrix(y_test, rm.predict(x_test))

# 분류 리포트 해석연습 다시. 디시전 트리 테스트가 더 잘 됨. 
print(classification_report(y_train, rm.predict(x_train))) # 100%
print(classification_report(y_test, rm.predict(x_test)))
    
엉망이다..ㅠㅠ

===
boosting
===

          adaptive~
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(n_estimators=20)
ada_model.fit(x_train, y_train)
accuracy_score(y_train, ada_model.predict(x_train))
accuracy_score(y_test, ada_model.predict(x_test))

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


   precision    recall  f1-score   support

           0       0.67      0.50      0.57        12
           1       0.45      0.62      0.53         8

    accuracy                           0.55        20
   macro avg       0.56      0.56      0.55        20
weighted avg       0.58      0.55      0.55        20

55%가 최고였다!!







