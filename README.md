# Machine-Learning-A-Z-Udemy

 ## Introduce

This course is [Machine Learning A-Z-Hands-On Python & R in Data Science](https://www.udemy.com/machinelearningchinese/) on Udemy. 
 There are Python and R code, while here I will only focus on the R .

1. I will create an R code file for each part. 
2. All DataSet can be downloaded [here](https://www.superdatascience.com/pages/%E4%B8%8B%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86)

This course is a practical course with vivid examples of practice, but the theory is usually less, 
only requires the high degree of mathematics. 
Most chapters use narrative to explain nouns, and then use sklearn tools to write  few code. 
 

In short, I feel the depth and amount of code are not enough, but it is still a good introductory course. 
If you have a basic theoretical foundation, 
this class should be very handy and easy to have a sense of accomplishment.


## About this course
- Part 1 - Data Preprocessing
- Part 2 - Regression: Simple Linear Regression, Multiple Linear Regression, Polynomial Regression
- Part 3 - Classification: Logistic Regression, SVM, Kernel SVM, Naive Bayes, Decision Tree, Random Forest
- Part 4 - Clustering: K-Means
- Part 5 - Association Rule Learning: Apriori
- Part 6 - Reinforcement Learning: Upper Confidence Bound, Thompson Sampling
- Part 7 - Natural Language Processing: Bag-of-words model and algorithms for NLP
- Part 8 - Deep Learning: Artificial Neural Networks, Convolutional Neural Networks
- Part 9 - Dimensionality Reduction: PCA, Kernel PCA
- Part 10 - Model Selection & Boosting: k-fold Cross Validation, Grid Search.
 
## R package and function

Spaling the dataset into Training Set and Test set
```
library(caTools)	
set.seed(123) 
split = sample.split(dataset$Salary,SplitRatio = 2/3)	
training_set <- subset(dataset , split == TRUE)
test_set <- subset(dataset , split == FALSE)
```
Feature Scaling
```
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])
```
Algorithm 

Algorithm     | Function     | Packages     
 -------- | :-----------:  | :-----------:  
Simple Learning Regression    |regressor= lm (formula = Salary ~ YearsExperience ,	data = training_set )     |     
Logistic    | classifier = glm(formula = Purchased ~ .,family= binomial,data= training_set)    |     
SVM    | classifier = svm(formula = Purchased ~ .,data = training_set,type = 'C-classification',kernel = 'linear')   | library(e1071)    | 3    |4       |5
Kernel SVM    | classifier = svm(formula = Purchased ~ .,data = training_set,type = 'C-classification',kernel = 'radial') #高斯核函數     | library(e1071)   | 3    |4       |5
Naive Bayes    | classifier =naiveBayes(x=training_set[-3],y=training_set$Purchased)     | library(e1071)    
Decision Tree    | classifier = rpart(formular = Purchased ~ . , data = training_set	)    | library(rpart)    
Random Forest    | classifier = randomForest(x = training_set[-3],y = training_set$Purchased,ntree=10)     | library(randomForest)   
K-means    | 第一列     | 第二列    
Apriori    | dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) <br> rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))   | library(arules)
Natural Language Processing    | corpus = VCorpus(VectorSource(dataset_original$Review)) #創建詞袋  <br> corpus = tm_map(corpus, content_transformer(tolower)) #大小寫轉換 <br> corpus = tm_map(corpus, removeNumbers) #清除數字  <br> corpus = tm_map(corpus, removePunctuation)  #清除標點 <br> corpus = tm_map(corpus, removeWords, stopwords()) #清除虛詞 <br> corpus = tm_map(corpus, stemDocument) # 詞根化 <br> corpus = tm_map(corpus, stripWhitespace) # 清洗空格 <br> dtm = DocumentTermMatrix(corpus) <br> dtm = removeSparseTerms(dtm, 0.999)| library(tm) <br> library(SnowballC) #清除英文虛詞    
Upper confidence Bound(UCB)   | 第一列     | 第二列 
Thompson   | 第一列     | 第二列 
Artificial Neural Networks   | 第一列     | 第二列    
PCA  | pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2) #pcaComp =把變量變成2個 <br> training_set = predict(pca, training_set)  <br> test_set = predict(pca, test_set) | library(caret)   
kernel PCA  | kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2) <br>  training_set_pca = as.data.frame(predict(kpca, training_set)) <br> training_set_pca$Purchased = training_set$Purchased   | library(kernlab)   


## Which classification algorithm should be used?


## summary of data mining package
Data mining mainly includes four categories: prediction, classification, clustering and association. Different mining purposes choose corresponding algorithms
### 連續因變量的預測：

- stats包 lm函數，實現多元線性回歸
- stats包 glm函數，實現廣義線性回歸
- stats包 nls函數，實現非線性最小二乘回歸-- lm是將曲線直線化再做回歸，nls是直接擬合曲線。需要三個條件：曲線方程、數據位置、係數的估計值。
ex nls(n~g/f(1+exp(u-t)/s)),data,start=c(h=8e4,u=16,s=5))$coef，其中h:高度,u:轉折點,s:離散程度
- rpart包 rpart函數，基於CART算法的分類回歸樹模型
- RWeka包 M5P函數，模型樹算法，集線性回歸和CART算法的優點
- adabag包 bagging函數，基於rpart算法的集成算法
- adabag包 boosting函數，基於rpart算法的集成算法
- randomForest包 randomForest函數，基於rpart算法的集成算法
- e1071包 svm函數，支持向量機算法
- kernlab包 ksvm函數，基於核函數的支持向量機
- nnet包 nnet函數，單隱藏層的神經網絡算法
- neuralnet包 neuralnet函數，多隱藏層多節點的神經網絡算法
- RSNNS包 mlp函數，多層感知器神經網絡
- RSNNS包rbf函數，基於徑向基函數的神經網絡

### 離散因變量的分類：
- stats包 glm函數，實現Logistic回歸，選擇logit連接函數
- stats包 knn函數，k最近鄰算法
- kknn包 kknn函數，加權的k最近鄰算法
- rpart包 rpart函數，基於CART算法的分類回歸樹模型
- adabag包bagging函數，基於rpart算法的集成算法
- adabag包boosting函數，基於rpart算法的集成算法
- randomForest包randomForest函數，基於rpart算法的集成算法
- party包ctree函數，條件分類樹算法
- RWeka包OneR函數，一維的學習規則算法
- RWeka包JPip函數，多維的學習規則算法
- RWeka包J48函數，基於C4.5算法的決策樹
- C50包C5.0函數，基於C5.0算法的決策樹
- e1071包svm函數，支持向量機算法
- kernlab包ksvm函數，基於核函數的支持向量機
- e1071包naiveBayes函數，貝葉斯分類器算法
- klaR包NaiveBayes函數，貝葉斯分類器算分
- MASS包lda函數，線性判別分析
- MASS包qda函數，二次判別分析
- nnet包nnet函數，單隱藏層的神經網絡算法
- RSNNS包mlp函數，多層感知器神經網絡
- RSNNS包rbf函數，基於徑向基函數的神經網絡

### 聚類：
- Nbclust包Nbclust函數可以確定應該聚為幾類
- stats包kmeans函數，k均值聚類算法
- cluster包pam函數，k中心點聚類算法
- stats包hclust函數，層次聚類算法
- fpc包dbscan函數，密度聚類算法
- fpc包kmeansruns函數，相比於kmeans函數更加穩定，而且還可以估計聚為幾類
- fpc包pamk函數，相比於pam函數，可以給出參考的聚類個數
- mclust包Mclust函數，期望最大（EM）算法

### 關聯規則：
- arules包apriori函數，Apriori關聯規則算法
