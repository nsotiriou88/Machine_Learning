# Apriori

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2)) #0.4 gives too many rules
# with irrelevant, high support products. Support = 3times*7days/7500allweeklytransactions. = 0.003.
# confidence = depends with alignment of future goals. Start with 0.8 and divide by 2.
summary(rules)

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
# Check about products with high support due from buyers, like mineral water.
# Decreasing the confidence, we can avoid including them in the results.
