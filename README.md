
# The portfolio during the ITWILL data-analyst course(Nov, 2020)

## (1) Analysis of the books data   

### Goal: make reading completed prediction algorithm model with the data purchased books and ML   

### [Process]  

### 1. collect data  
-collect personal data(input 100 books that I recently have bought) + scrap regarding data(yes24, book shopping mall, bestseller top 100)  
-> total 200 books  

### 2. data cleansing  
-change column names  
-change data type(one-hot-encoding, categorize)  

### 3. data preprocessing and select features  
-data normalization & standardization  
-select feature that would be assumed to meaningful  

### 4. check the important columns through executing ML and visualization  
-Decision Tree  
-Ensemble(bagging, random forest, boosting)  
-1)Train, Test: execute with my personal data 2) Train: the personal data Test: the scrapped data  

### 5. derive insights from the result

![selfyesrandom](https://user-images.githubusercontent.com/66956480/105605757-9b511200-5dd9-11eb-8a77-5ca7a92056f1.png)  

-test with yes24 data(100 books, random results of whether it has read completely or not) after train with self-made data(100 books)  
-confirm prediction accuracy between 50% and 57%  
-important columns are the weight of the book, how many pages of the book, salespoint, how many reviews, in order.  

## (2) Analysis of youtube video comments of the publisher with Kmeans clustering  

### Intention: clustering comments from youtube videos of publishers that are popular based on the counts of comments after pre-processing, which is to confirm a hypothesis that the type of comments would be different publisher’s comments one another.  

### [Process]  

### 1. collect data  
-select three youtube videos, collect them with selenium module  

### 2. data cleansing & pre-processing  
-concatenate in a data frame and execute Natural Language Processing(NLP, with regular expression)  

### 3. clustering with Kmeans module   
-measure with Siluette and Elbow method.  
-visualizing with Kmeans based on the result from the measurement by S & M  

### 4. come to the conclusion  
-the result showed that the adequate counts for the clustering are two.  
-It was not right what I expected it would be different the type of comments each publisher one another.  

![kmeansvisual](https://user-images.githubusercontent.com/66956480/105605863-c8052980-5dd9-11eb-960f-25ce80ed56a0.png)  


## (3) Analysis of youtube video comments of the publisher with association rules   

### Intention: executing association rules of comments from youtube videos of publishers that are popular based on the counts of comments after pre-processing, and try to find the marketing point from the insight that  

### [Process]  

### the first, second stages are the same as the Kmeans clustering’s  

### 3. execute analysis with apriori method  

### 4. visualize the result and find some points, go to a conclusion  
It was found that we could make use of the word that appeared from the comments in some marketing points.  

![assorules](https://user-images.githubusercontent.com/66956480/108652128-bca33c00-7506-11eb-974b-b466940bf25e.png)  





# 도서 데이터 분석

## 기획의도 : 도서 구매 관련 데이터와 머신러닝을 이용해 완독 예측 알고리즘 모델을 만든다. 

## 작업 과정 

### 1. 데이터 수집
-개인 데이터 수집(최근 구매한 도서 100권 정보 입력)
-외부 데이터 수집(yes24 베스트셀러 100위)

이미지

### 2. 데이터 클리닝
-컬럼명 변경  
-데이터 형 변경(원 핫 인코딩, 범주화 등)

### 3. 데이터 전처리 및 피처 선정
-데이터 표준화 및 정규화  
-유의미할 것으로 추정되는 컬럼 피처로 선정

### 4. 머신러닝 수행 및 시각화를 통한 중요 컬럼 확인 
-Decision Tree  
-Ensemble(bagging, random forest, boosting)  
-1)Train, Test : 개인 데이터로 나누어 진행 2) Train : 개인 데이터 Test : 외부 수집 데이터

### 5. 결과 바탕으로 인사이트 도출
-자체 제작 데이터(100권)로 트레이닝 후 yes24 베스트셀러 데이터(100권, 완독여부 랜덤 삽입)테스트 진행  
-50% ~ 57% 사이의 예측 정확도 확인  
-중요 컬럼은 책 무게(wesn), 페이지 수(pasn), 세일즈포인트(spsn), 리뷰 갯수(rsn) 순으로 나타남

![selfyesrandom](https://user-images.githubusercontent.com/66956480/105605757-9b511200-5dd9-11eb-8a77-5ca7a92056f1.png)

# 출판사 댓글 군집 분석

## 기획의도 : 문학 관련 일정 규모 이상의 출판사 세 곳의 인기 유튜브 영상 댓글을 가공해 군집화 한다. 출판사 별 댓글 유형이 다를 것이라는 가설을 검증하기 위함.

## 작업과정

### 1. 데이터 수집
-유튜브 영상 세 개 선정, 각각 수집(selenium활용)

### 2. 데이터 클리닝 및 전처리 
-데이터 프레임으로 합친 후 자연어 처리(특수문자 등 불필요한 부분 제거 후 진행)

### 3. 메소드 활용하여 군집화(kmeans, siluette, elbow)
-엘보우, 실루엣 메소드 활용해 적당한 군집 수 측정  
-군집 수 감안하여 kmeans 메소드 활용, 시각화 및 확인.

### 4. 결론 도출
-적당한 군집 수는 두 개로 나타남  
-출판사 별로 댓글 작성 유형이 다를 것으로 예측했으나 맞지 않음. 

![kmeansvisual](https://user-images.githubusercontent.com/66956480/105605863-c8052980-5dd9-11eb-960f-25ce80ed56a0.png)

# 출판사 댓글 연관성 분석 

## 기획의도 : 문학 관련 일정 규모 이상의 출판사 세 곳의 인기 유튜브 영상 댓글 분류를 통해 특징을 파악하고 마케팅 포인트를 찾아본다. 

## 작업 과정

### 1, 2번은 군집 분석과 동일

### 3. 메소드 활용하여 연관성 분석(apiori)

### 4. 분석 결과 시각화 하여 특징 찾기, 결론 도출
-댓글에서 자주 나온 단어를 확인해 마케팅 포인트로 삼을 수 있음을 확인

![assorules](https://user-images.githubusercontent.com/66956480/108652128-bca33c00-7506-11eb-974b-b466940bf25e.png)

## [발표용 자료](https://www.notion.so/Pdf-file-for-presentaion-98595c70379241f89df77d19bb5ee6c7)

