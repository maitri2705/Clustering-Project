# Clustering-Project

1. lloyd.py : Lloyd's k-means.
2. kmeanspp.py : k-means++.
3. cmeans.py : fuzzy c-means.

Notice that the fuzzy c-means algorithm should be implemented to produce hard clustering. This should be done by first computing soft clustering and then converting the soft clustering to hard clustering.

Input1: One data file. The data is a comma separated matrix of size n  m. Here the data points are the rows, not the columns.
Input2: k, the number of desired clusters.
Input3: r, the number of random iterations.
Output: A comma separated file containing n integer values. Each value is in the range 0...k-1.
