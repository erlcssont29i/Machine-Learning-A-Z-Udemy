
Part_2: 回歸 Regression

回归模型（线性或非线性）被广泛应用于数值预测，比如薪水，销售额等等。
如果说自变量（independent variable）是时间，那么我们是在预测未来的数值；反之我们的模型在预测当前未知的数值。

	1. 簡單線性回歸 	(Simple Learning Regression)
	2. 多元限性回歸 	(Multiple Learning Regression)
	3. 多項式回歸 	(Polynomial Regression)



==============Simple Linearning Regression===================
 *** Data說明：30個員工，對每一個員工的"工作年數"與"工資" ***
 	=> 問題是：**用工齡(自變量）預測工資（應變量)**


# mporting the Dataet
dataset<- read.csv('Salary_Data.csv')
# dataset <- dataset[ , 2:3]

# Spaling the dataset into Training Set and Test set)	
library(caTools)	
set.seed(123) 
split = sample.split(dataset$Salary,SplitRatio = 2/3)	
training_set <- subset(dataset , split == TRUE)
test_set <- subset(dataset , split == FALSE)

# Feature Scaling
# training_set[,2:3] =scale (training_set[,2:3]) # X must be numeric ,只對第二行＆第三行(年齡&薪水)進行特徵縮放
# test_set[, 2:3] =scale (test_set[,2:3])


# Fitting Simple Learning Regression to the Training Set
regressor= lm (formula = Salary ~ YearsExperience ,
				data = training_set )
	# summary(regressor) -- p-value越小越顯著，一般"<0.05即有顯著相關"

# Predicting the Test set results
	# 利用已經擬合好的回歸器對測試集當中的數據做預測，對預測結果儲存在y_prde，可以對測試集的實際數字作比較
y_pred = predict (regressor , newdata = test_set)


# Visualising the Training set sesults
ggplot() + 
  geom_point (aes(x = training_set$YearsExperience,y = training_set$Salary),
             colour = 'red') +
  geom_line (aes(x = training_set$YearsExperience, y = predict (regressor , newdata = training_set)),
            colour = 'blue') +
  ggtitle ('Salary vs YearsExperience (Training set)' ) +
  xlab ('Years of Experience') +
  ylab ('Salary')



# Visualising the Test set sesults
ggplot() + 
  geom_point (aes(x = test_set$YearsExperience,y = test_set$Salary),
              colour = 'red') +
  geom_line (aes(x = training_set$YearsExperience, y = predict (regressor , newdata = training_set)),
             colour = 'blue') +
  ggtitle ('Salary vs YearsExperience (Test set)' ) +
  xlab ('Years of Experience') +
  ylab ('Salary')



 ==============Multiple Linearning Regression===================

 *** Data說明：50個初創公司 共5個變數，其中應變量是profit 自變量是前面四個變數***

 *** Assumptions of a Linear Regression ***
	 1. Linearity 					線性
	 2. Homoscedasticity 			同方差性
	 3. Multivariate normality 		多元正態分佈
	 4. Independence or errors 		獨立誤差
	 5. Lack of multicollinearity 	無多重共綫性

 *** Dummy Variables ***
 	把state in ('New York','California') 變成 Ｎew York in (0,1)
 		=>兩個變量=一個虛擬變量 # y=b0+b1*X1+b2*X2+b3*X3+b4*D1

 	虛擬變量的陷阱：違反多重共線性 Always omit one dummy variable


*** Building A Model (step by step) ***
如何決定變量X是否應該進入Ｙ，如果沒顯著相關則不應該用其解釋

five methods of Building Models
	1. All-In 
	2. Backwrd Elimination 			反淘汰
	3. Forward Selection 			順向選擇
	4. Bidirectionl Elimination 	雙向淘汰
	5. Score Comparison 			信息量比較


# mporting the Dataet
dataset<-read.csv('50_Startups.csv')

dataset$State = factor(dataset$State ,
							levels =c ('New York','California','Florida'),
							labels=c(1,2,3) )

# Spaling the dataset into Training Set and Test set)
library(caTools)	
set.seed(123) 
split = sample.split(dataset$Profit, SplitRatio = 0.8 )	
training_set <- subset(dataset , split == TRUE)
test_set <- subset(dataset , split == FALSE)

# Feature Scaling
# training_set[,2:3] =scale (training_set[,2:3]) # X must be numeric ,只對第二行＆第三行(年齡&薪水)進行特徵縮放
# test_set[, 2:3] =scale (test_set[,2:3])


# install.packages('caTools)
regressor = lm ( formula = Profit ~. , 
                 data = training_set)
summary(regressor)

# Predicting The test set results
	# 利用已經擬合好的回歸器對測試集當中的數據做預測，對預測結果儲存在y_prde，可以對測試集的實際數字作比較
y_pred= predict (regressor ,newdata=test_set)



*** Backwrd Elimination  反淘汰 ***
	1. Select a significance level to stay in then model (ex: SL = 0.05)
	2. Fit the Full model with all posible predictors
	3. Consider the predictor with the *highest* P-value. IF P>SL ,go to STEP 4 ,otherwise go to FIN
	4. Remove the predictor
	5. Fit model without this variable


# Building the optimal model using Backward Elimination 
regressor = lm ( formula = Profit ~R.D.Spend + Administration + Marketing.Spend +State , 
                 data = training_set)
summary(regressor)

regressor = lm ( formula = Profit ~R.D.Spend + Administration + Marketing.Spend ,  # Remove the predictor
                 data = training_set)
summary(regressor)

regressor = lm ( formula = Profit ~R.D.Spend  + Marketing.Spend ,  # Remove the predictor
                 data = training_set)
summary(regressor)

regressor = lm ( formula = Profit ~R.D.Spend,  # Remove the predictor
                 data = training_set)
summary(regressor)




 ==============Polynomial Linear Regression===================
 # y=b0+b1*x1+b2*x1^2+....+bn*X1^n

 *** Data說明：3個變量：職業、級別、薪資。10個不同職位、級別對應的平均薪資***


# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# 這個案例不需要分成訓練集/測試集
		# Splitting the dataset into the Training set and Test set
		# install.packages('caTools')
		# library(caTools)
		# set.seed(123)
		# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
		# training_set = subset(dataset, split == TRUE)
		# test_set = subset(dataset, split == FALSE)

# 這個案例不需要特徵縮放
		# Feature Scaling
		# training_set[, 2:3] = scale(training_set[, 2:3])
		# test_set[, 2:3] = scale(test_set[, 2:3])


# Fitting Linear Regression to the dataset
lin_reg=lm(formula = Salary~ Level,
           data=dataset)

summary(lin_reg)

			Residuals:
			    Min      1Q  Median      3Q     Max 
			-170818 -129720  -40379   65856  386545 

			Coefficients:
			            Estimate Std. Error t value Pr(>|t|)   
			(Intercept)  -195333     124790  -1.565  0.15615   
			Level          80879      20112   4.021  0.00383 **


# Fitting Polynomial Regression to the dataset
dataset$level2 = dataset$Level^2
dataset$level3 = dataset$Level^3

poly_reg = lm(formula = Salary~.,
              data=dataset)

summary(poly_reg) # 新添加的level2 and level3 都有顯著性

		Residuals:
		   Min     1Q Median     3Q    Max 
		-75695 -28148   7091  29256  49538 

		Coefficients:
		             Estimate Std. Error t value Pr(>|t|)   
		(Intercept) -121333.3    97544.8  -1.244  0.25994   
		Level        180664.3    73114.5   2.471  0.04839 * 
		level2       -48549.0    15081.0  -3.219  0.01816 * 
		level3         4120.0      904.3   4.556  0.00387 **


 *** 畫圖看看線性迴歸＆多項式回歸的差異***

#Visualising the Linear Regression results

	#install.packages('ggplot2')
	#library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour = 'red') +
  geom_line (aes (x=dataset$Level, y = predict (lin_reg, newdata=dataset)),
             colour = 'blue') +
  xlab('Level') +
  ylab('Salary')

#Visualising the Polynomial Regression results
ggplot() +
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour = 'red') +
  geom_line (aes (x=dataset$Level, y = predict (poly_reg, newdata=dataset)),
             colour = 'blue') +
  xlab('Level') +
  ylab('Salary')


 ============== ============== ============== ============== ==============
 *** 自己練習：把線性迴歸＆多項式回歸的圖形化在一張圖，來看看差異***		
																				
ggplot() +
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour = 'red') +
  geom_line (aes (x=dataset$Level, y = predict (lin_reg, newdata=dataset)),
             colour = 'blue') +
  geom_line (aes (x=dataset$Level, y = predict (poly_reg, newdata=dataset)),
             colour = 'yellow') +

  # scale_colour_manual("",values = c("Respiratory" = "red","COPD" = "green"))+
  xlab('Level') +
  ylab('Salary')

============== ============== ============== ============== ==============

#Predicting a new result with Linear Regression
y_pred=predict(lin_reg, data.frame(Level=6.5))

#Predicting a new result with Polynomial Regression
y_pred=predict(poly_reg, data.frame(Level=6.5,
                                    level2=6.5^2,
                                    level3=6.5^3,
                                    level4=6.5^4))




*** R2 & Adjusted R2*** 
	丟入越多自變量，R2月高，模型解釋性越強，這是不合理的，所以用Adjusted R2來修正R2


