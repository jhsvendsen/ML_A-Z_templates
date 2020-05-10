# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')

X <- dataset[4:5]

# Using elbow method to find optimal # of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)

plot(1:10,wcss, type = 'b', main = paste('Clusters of clients'), xlab = 'Number of clusters', ylab = 'wcss')

# Applying k-means to the mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualising the data
library(cluster)
clusplot(X,
          kmeans$cluster,
          lines = 0,
          shade = TRUE,
          color = TRUE,
          labels = 2,
          plotchar = FALSE,
          main = paste('Cluster of clients'),
          xlab = 'Annual Income',
          ylab = 'Spending score')