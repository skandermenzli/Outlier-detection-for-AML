# Statistical methods for outlier detection

Date Created: July 14, 2023 10:50 AM
Status: Doing

# Stats 101:

### Mean:

Also known as the averge is the sum of all values in the dataset divided by their total number, it summarizes the entire dataset with a single number providing a measure of central tendency that represents typical value of the data

![download.png](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/download.png)

### Standard deviation:

It’s a statistical measure that quantifies the amount of variability in the dataset, basicly it measures how much data points deviate on average from the mean.

![download.png](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/download%201.png)

### Median:

The median is a statistical measure that represents the middle value of a dataset when it is arranged in ascending or descending order. It is a measure of central tendency that is not influenced by extreme values or outliers.

### Quartiles:

In statistics quartiles are value that divided our data into 4 parts each with an equal number of observations, they provide insights into the spread of data and the range of their values.

there are 3 quatile values:

- Q1 (the lower quartile): the middle number between the smallest value and the median,represents the 25th percentile i.e 25% of the data is below it
- Q2 (the median): 50% of data points are below it
- Q3 (upper quartile): point between median and highest value, 75% of data points are below it

### Interquartile Range (IQR):

The interquartile range (IQR) is a measure derived from quartiles and represents the spread of the central 50% of the data. It is calculated as the difference between Q3 and Q1: IQR = Q3 - Q1

![Interquartile-Even.png](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/Interquartile-Even.png)

# Statistical methods for outlier detection:

Now we will look at some univariate methods for outlier detection, these methods identifiy anomalies based on the analysis of a single variable, they include:

- Z-score
- Modified Z-score
- Tukey’s fences
- Grubb’s test

# Z-score:

the Z-score is a statistical measure that indicates how many standard deviation away from the mean of a distribution a data point is. it’s calulated by this formula:

![zscore-56a8fa785f9b58b7d0f6e87b.gif](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/zscore-56a8fa785f9b58b7d0f6e87b.gif)

The z-score assumes the data follws a normal distribution and Applying the above formula  to every point in our dataset simply converts it to standard normal distribution with mean 0 and stanadard deviation of 1. We can then use the Z-table to tell which percentage of values falls below  acerain z-score:

![8573955.webp](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/8573955.webp)

using this we can set a threshold to determine wether a point is an outlier or not. a common threshold is  z< -2 or z>2 i.e points that are more than 2 standard deviations away from the mean since 95% of points fall in that range we can consider any outside it as outliers.

![z-scores-formula-concepts-and-examples.jpg](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/z-scores-formula-concepts-and-examples.jpg)

# Modified Z-score:

The modified Z-score is a variation of the regular Z-score that is more robust to outliers. Since the regular uses the mean and standard deviation that are themselves susceptible to anomilies, for example if our data containes one extreme outlier it can significantly alter the value of the mean thus many anomilies will be uncaught.

The modified z-score adressses this bu using the median and the median absolute deviation(MAD)

![0_L_rMyBjsL1aLnRSC.png](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/0_L_rMyBjsL1aLnRSC.png)

we  multiply by 0.6745 to ensure that the modified z-score has a similar range to the values of the standard z-score

# Tukey's Fences:

Tukey’s fences is a statistical methode for outlier detection, it defines a range where most of the data is expected to fall and any point outside it is considered an outlier. All we need to do is :

- sort our data in ascending order
- calculate the IQR=Q1-Q3
- the lower fence(bound) = Q1 - 1.5*IQR
- the upper fence(bound) = Q3 + 1.5*IQR

any data point below the lower fence or above the upper fence is an outlier. the 1.5 factor is commonly used but we can adjust it to change our bounds

**Assumption: t**his method is based on the assumption that our data follows a roughly symmetric distribution that’s why it might not be suitable for skewed or non-normal distributions. it also assumes that data points are independent and not influenced by each other

# Grubbs' test:

Grubb’s test is a statistical method used for detecting one outlier in univariate dataset, the test assumes that our data folows a normal distribution. Once an outlier has been detected we can remove it and repeat the process itervatly to detect more than one outlier. The alogrithm is as follows:

1 we calculate the G statistic :

![Untitled](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/Untitled.png)

2 then we compare it to the critical G value at the desired significance level if it’s higher than the point will be marked as an outlier, we can obtain the the critical value by this formula:

![grubbs-4.webp](Statistical%20methods%20for%20outlier%20detection%205311f4b115e34e6c91581bd750d669fd/grubbs-4.webp)

**Assumption:** the grubbs’s test assumes the data follows a normal distribution if not the results will be less reliable