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
Natural Language Processing    | corpus = VCorpus(VectorSource(dataset_original$Review)) # 創建詞袋  <br> corpus = tm_map(corpus, content_transformer(tolower)) #大小寫轉換 <br> corpus = tm_map(corpus, removeNumbers) #清除數字  <br> corpus = tm_map(corpus, removePunctuation)  #清除標點 <br> corpus = tm_map(corpus, removeWords, stopwords()) #清除虛詞 <br> corpus = tm_map(corpus, stemDocument) # 詞根化 <br> corpus = tm_map(corpus, stripWhitespace) # 清洗空格 <br> dtm = DocumentTermMatrix(corpus) <br> dtm = removeSparseTerms(dtm, 0.999)| library(tm) <br> library(SnowballC) #清除英文虛詞    
Artificial Neural Networks   | 第一列     | 第二列    
PCA  | pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2) #pcaComp =把變量變成2個 <br> training_set = predict(pca, training_set)  <br> test_set = predict(pca, test_set) | library(caret)   
kernel PCA  | kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2) <br>  training_set_pca = as.data.frame(predict(kpca, training_set)) <br> training_set_pca$Purchased = training_set$Purchased   | library(kernlab)   


## Which classification algorithm should be used?

