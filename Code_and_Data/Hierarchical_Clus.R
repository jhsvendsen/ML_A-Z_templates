# Heirachel clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using dendrogram to find opt. nr. of clusters
dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fit model
hc = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the data
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         main = paste('Cluster of clients'),
         xlab = 'Annual Income',
         ylab = 'Spending score')