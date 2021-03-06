
Part_1:數據預處理

================================

# 1.下載數據集 (Get the Dataset)
# 2.導入標準庫 (IMporting the Libraries) --  即導入R包
# 3.導入數據集 (Importing the Dataet)
dataset <- read.csv('Data.csv')

# 4.缺失數據 (Takeing Care of Missing Data)
	# 把'Salary' and 'Age'的缺失數據用平均值替代
	# 也可以嘗試用中位數/眾數
dataset$Age[is.na(dataset$Age)] = mean(dataset$Age, na.rm = T) # na.rm表示告訴mean函數remove NULL
dataset$Salary[is.na(dataset$Salary)] = mean(dataset$Salary, na.rm = T)

# 5.分類數據 (Encoding categorical data)
	# 把country轉換成數據、把是否購買 Y/N 轉換成 1/0
	# 因子(factor)即表示類別
dataset$Country = factor(dataset$Country ,
							levels =c ('France','Spain','Germany'),
							labels=c(1,2,3) )

dataset$Purchased = factor(dataset$Purchased ,
							levels =c ('No','Yes'),
							labels=c(0,1) )

# 6.將數據分成訓練集&測試集 (Spaling the dataset into Training Set and Test set)
	# 將訓練集的結果，丟到測試集來判斷 (用訓練集擬合模型，用測試集來判斷模型性能)
	
set.seed(123) # 隨機數組生成的方式
split = sample.split(dataset$Purchased,SplitRatio = 0.8 )	# 將數據分成訓練集&測試集
training_set <- subset(dataset , split == TRUE)
test_set <- subset(dataset , split == FALSE)

# 7.特徵縮放 (Feature Scaling)
	# Age 跟 Salary的數字不在同一個量級-->薪水數量級太大(相對年齡)，造成模型主要受到薪水的影響
	# 將不同數量級的變量，縮放到同一個數量級
	# 特徵縮放的方法： 
		#1.標準化(Standardisation) =x是一個平均值為0，標準差為1的分佈 。
		#2.正常化(Normalisation) = X是0~1之間的小數

training_set[,2:3] =scale (training_set[,2:3]) # X must be numeric ,只對第二行＆第三行(年齡&薪水)進行特徵縮放
test_set[, 2:3] =scale (test_set[,2:3])
	View(training_set)
	View(test_set)

==============數據預先處理模板 (Data Perprocessing Templet)==============

# mporting the Dataet
dataset<- read.csv('Data.csv')
# dataset <- dataset[ , 2:3]

# Spaling the dataset into Training Set and Test set)
library(caTools)	
set.seed(123) 
split = sample.split(dataset$DependentVariable, SplitRatio = 0.8 )	
training_set <- subset(dataset , split == TRUE)
test_set <- subset(dataset , split == FALSE)

# Feature Scaling
# training_set[,2:3] =scale (training_set[,2:3]) # X must be numeric ,只對第二行＆第三行(年齡&薪水)進行特徵縮放
# test_set[, 2:3] =scale (test_set[,2:3])




