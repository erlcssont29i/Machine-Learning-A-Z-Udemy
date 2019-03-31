Part_7: 自然語言處理 Natural Language Processing
Q:預測評論是正面還是負面

	Step1: 讀取數據
	Step2: 創建詞袋
	Step3: 大小寫轉換
	Step4: 清理數字
	Step5: 清理標點
	Step6: 清理虛詞
	Step7: 詞根化
	Step8: 清洗多餘空格
	Step9: 建立稀疏矩陣
	Step10: 最後一步

貝氏、決策樹、隨機森林 更加適合自然語言處理(經驗之談)
本例子使用隨機森林


# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')

**tm包是R文本挖掘方面不可不知也不可不用的一个package。它提供了文本挖掘中的综合处理功能。
** 如：数据载入，语料库处理，数据预处理，元数据管理以及建立“文档-词条”矩阵。
** tm包介紹：http://yphuang.github.io/blog/2016/03/04/text-mining-tm-package/
library(tm)
library(SnowballC) #清除英文的虛詞
corpus = VCorpus(VectorSource(dataset_original$Review)) # 創建詞袋
corpus = tm_map(corpus, content_transformer(tolower)) #大小寫轉換 （map=mapping 映射)
# as.character(corpus[[1]])

corpus = tm_map(corpus, removeNumbers) #清除數字，因為數字對目的不重要，相當於清除噪音
# as.character(corpus[[841]])

corpus = tm_map(corpus, removePunctuation)  #清除標點

# install.packages('SnowballC')
corpus = tm_map(corpus, removeWords, stopwords()) #清除英文的虛詞

corpus = tm_map(corpus, stemDocument) # 詞根化
corpus = tm_map(corpus, stripWhitespace) # 清洗多餘空格

# Creating the Bag of Words model ＝將文本從corpus中轉化為一個龐大的稀疏矩陣
dtm = DocumentTermMatrix(corpus)
> dtm
<<DocumentTermMatrix (documents: 1000, terms: 1577)>>
Non-/sparse entries: 5435/1571565 # 不是0的個數/0的個數
Sparsity           : 100% #稀疏性，太稀疏了
Maximal term length: 32
Weighting          : term frequency (tf)

dtm = removeSparseTerms(dtm, 0.999) # 過濾 優化的步驟 把出現次數只有0.001的（1000列只出現一次的word）排除，目的是簡易化運算

<<DocumentTermMatrix (documents: 1000, terms: 691)>>
Non-/sparse entries: 4549/686451
Sparsity           : 99%
Maximal term length: 12
Weighting          : term frequency (tf)


dataset = as.data.frame(as.matrix(dtm)) #要使用隨機森林算法，數據結構是資料匡，所以轉換一下
dataset$Liked = dataset_original$Liked # 加入應變量

=======================
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)

