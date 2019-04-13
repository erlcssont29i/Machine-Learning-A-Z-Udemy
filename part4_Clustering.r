Part_4: 聚類 Clustering
K-means為經典的聚類算法，因為: 1.直覺容易理解，但數學其實不簡單、2. 運算快

** Question:
1. 如何確定類的個數
2. 如何確認要分到哪一個類別

Step1: Choice the nunber K of clusters
Step2: Select at random K points, the centroids (not necessarily from your data)
Step3: Assign each data point to closest centroid -->that forms K clusters
Step4: Compute and place the new centroid of each cluster
Step5: Reassing each data point to the new closest centroid.
		If any reassignment took place, go to STEP 4, otherwise go to FIN. 

** 初始中心的陷阱(Random Initialization Trap)
初始中心點的選擇對K-means的影響:初始中心點不一樣，得到的結果不一樣-->即初始中心點不該是隨機選擇的-->優化模型成'k-means++'

** Choosing the right number of clusters
組內平方和WSCC: 隨著K增加(組數)，WSCC越小-->在用手肘法則(The Elbow Method)選擇最適合的K


# K-Means Clustering

# Importing the dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of cluster',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans = kmeans(X, 5, iter.max =300, nstart =10)
y_kmeans = kmeans$cluster

# Visualising the clusters
library(cluster)
clusplot(X,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels =2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')


