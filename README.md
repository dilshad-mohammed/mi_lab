# mi_lab

# 1-) Perceptron

# def stepfunction(z): 
#     if z>=0:
#         return 1 
#     else:
#         return 0 
# w=[0.01,0.01] 
# b=0.01 
# l_r=0.001                        #l_r=learning rate 
# x=[0,0,1,1] 
# y=[0,1,0,1] 
# z=[0,1,1,1] 
# for epoc in range(20):
#     i = 0 
#     while i <= 3: 
#         print('no of epocs', epoc)
#         print('x:', x[i], 'y:', y[i]) 
#         z1= stepfunction(w[0] * x[i] + w[1] * y[i] + b) 
#         print('output:',z1) 
#         E = z[i] - z1 
#         print('Error:',E) 
#         w[0] = w[0] + l_r * E * x[i] 
#         print('weight_1n:',w[0]) 
#         w[1] = w[1] + l_r * E * y[i] 
#         print('weight_2n:',w[1]) 
#         b = b + l_r * E 
#         print('b_n:',b)         #b_n=bayes new 
#         i += 1


# 2-) Linear Regression

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# df=pd.read_csv(r'C:\Users\MyPC\Desktop\dilshad\DUK\MI\Lab_Assig\lab files\salary_data.csv')
# def findYprediction(data):
#     x = data['YearsExperience'].values
#     # print(f"X={x}")
#     y = data['Salary'].values
#     x_mean = np.mean(x)
#     # print(f"mean of x: {x_mean}")
#     y_mean = np.mean(y)
#     # print(f"mean of y: {y_mean}")
#     beta_1 = ( np.sum((x-x_mean)*(y-y_mean)) ) / ( np.sum((x-x_mean) ** 2))
#     # print(f"Beta 1: {beta_1}")
#     beta_0 = y_mean - beta_1 * x_mean
#     # print(f"Beta 0: {beta_0}")
#     Y = beta_0 + beta_1 * x
#     return Y
# y = findYprediction(df)
# print(y)
# plt.scatter(df.YearsExperience, df.Salary, color="blue")
# plt.plot(list(df.YearsExperience), y, color="red")
# print(plt.show())

# 3-) Logistic Reression

# import pandas as pd
# import numpy as np
# data = {
#     'medication dosage': [20, 30, 40, 50],
#     'cured': [1, 2, 4, 6],
#     'total patients': [5, 6, 6, 7]
# }
# df = pd.DataFrame(data)
# # print(df.head())
# df['probability'] = df['cured']/df['total patients']
# df['odds'] = df['probability']/(1-df['probability'])
# df['logit'] = np.log(df['odds'])
# def linearregressionfunct(x,y):
#     x_mean = np.mean(x)
#     y_mean = np.mean(y)
#     beta_1 = (np.sum((x-x_mean)*(y-y_mean)))/(np.sum((x-x_mean)**2))
#     beta_0 = y_mean-beta_1*x_mean
#     return beta_0,beta_1

# x = df['medication dosage'].values
# y = df['logit'].values

# beta_0,beta_1 = linearregressionfunct(x,y)

# def predictprobability(dossage):
#     logit = beta_1*dossage+beta_0
#     print('Logit:',logit)
#     probability = 1/(1+np.exp(-logit))
#     return probability

# dossage = int(input('Enter a Dossage :'))
# if predictprobability(dossage)<0.5:
#     print('Patient is not Cured')
# else:
#     print('Patient is Cured')

x_range = np.linspace(min(x), max(x), 100)  
y_pred = beta_0 + beta_1 * x_range  
y_prob = 1 / (1 + np.exp(-y_pred)) 

plt.scatter(df['medication dosage'], df['probability'], color='blue', label='Data points')
plt.plot(x_range, y_prob, color='red', label='Logistic Regression Curve', linewidth=2)
plt.show()

# 4-) Backpropogation XOR

# import numpy as np
# import math
# import random

# # XOE dataset
# data = [
#     ([0, 0], 0),
#     ([0, 1], 1),
#     ([1, 0], 1),
#     ([1, 1], 0)
# ]

# alpha = 0.5             #learning_rate=alpha

# np.random.seed(42)
# # initialize the weights
# w11, w12 = np.random.uniform(-1,1,(2, ))
# w21, w22 = np.random.uniform(-1,1,(2, ))

# # initialize the output weights
# w13, w23 = np.random.uniform(-1,1,(2,  ))

# # Sigmoid activation function and its derivative
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Train the network
# epochs = 10000
# for epoch in range(epochs):
#     total_error = 0
#     for inputs, actual in data:
#         x1, x2 = inputs

#         # forward propogation pass through hideen layer
#         y_1 = x1 * w11 + x2 * w21
#         a_1 = sigmoid(y_1)

#         y_2 = x1 * w12 + x2 * w22
#         a_2 = sigmoid(y_2)

#         # forward propogation pass through output layer
#         y_3 = a_1 * w13 + a_2 * w23
#         a_3 = sigmoid(y_3)

#         # calc. error
#         error = actual - a_3
#         total_error += error ** 2                  # Summing up squared errors for each instance

#         # Backpropagation for output layer weights
#         a__3 = sigmoid_derivative(a_3)

#         # Update output layer weights
#         delta_w13 = alpha * error * 1 * a_1
#         delta_w23 = alpha * error * 1 * a_2

#         w13 += delta_w13
#         w23 += delta_w23

#         # Backpropagate the error to the hidden layer
#         error_y1 = error * a__3 * w13
#         error_y2 = error * a__3 * w23

#         # Derivatives at hidden neurons
#         a__1 = sigmoid_derivative(a_1)
#         a__2 = sigmoid_derivative(a_2)

#         # Calculate weight updates for hidden layer weights
#         delta_w11 = alpha * error_y1 * a__1 * x1
#         delta_w12 = alpha * error_y2 * a__2 * x1
#         delta_w21 = alpha * error_y1 * a__1 * x2
#         delta_w22 = alpha * error_y2 * a__2 * x2

#         # Update hidden layer weights
#         w11 += delta_w11
#         w12 += delta_w12
#         w21 += delta_w21
#         w22 += delta_w22

#     total_error /= len(data)
#     # Print total error every 1000 epochs for tracking
#     if epoch % 1000 == 0:
#         print(f"Epoch {epoch}, Total Error: {total_error}")

# # Test the network after training
# print("\nTesting the network after training:")
# for inputs, actual in data:
#     x1, x2 = inputs
#     # Forward pass
#     y1 = x1 * w11 + x2 * w21
#     a_1 = sigmoid(y1)
#     y2 = x1 * w12 + x2 * w22
#     a_2 = sigmoid(y2)
#     y_3 = a_1 * w13 + a_2 * w23
#     a_3 = sigmoid(y_3)
#     # Output the results
#     print(f"Input: {inputs}, Predicted Output: {round(a_3)}, Actual Output: {actual}")

# 5-) Mle

# data = [1, 0, 1, 1, 0, 1, 1, 0, 0, 1]

# infected = sum(data)
# n = len(data)

# mle = infected / n
# print(mle)

# 6-) K-means

# X = np.random.rand(1000, 2)
# k = 6

# def euclidean_distance(a, b):
#     return np.sqrt(np.sum((a - b)**2, axis=1))

# # k-means algorithm implementation from scratch
# def kmeans(X, k, max_iters=100):
#     # Randomly initialize centroids by selecting k random points from the dataset
#     centroids = X[np.random.choice(X.shape[0], k, replace=False)]
#     iterations = 0

#     for i in range(max_iters):
#         iterations += 1

#         labels = np.array([np.argmin(euclidean_distance(x, centroids)) for x in X])
#         new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

#         if np.all(centroids == new_centroids):
#             break
#         centroids = new_centroids
#     return labels, centroids, iterations

# # Apply K-Means clustering
# labels, centers, total_iterations = kmeans(X, k)

# # Print the total number of iterations
# print(f"K-means converged in {total_iterations} iterations.")

# # Create a scatter plot of the data points
# fig, ax = plt.subplots()
# ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

# # Mark the cluster centers with 'x'
# ax.scatter(centers[:, 0], centers[:, 1], marker='x', s=400, linewidths=3, color='r')

# plt.show()

# 7-) Decision Tree(scratch)

# df = pd.read_csv(r"C:\Users\MyPC\Desktop\dilshad\DUK\MI\Lab_Assig\lab files\decision tree.csv")

# encoding = {
#     "age": {"<=30": 0, "31...40": 1, ">40": 2},
#     "income": {"low": 0, "medium": 1, "high": 2},
#     "student": {"no": 0, "yes": 1},
#     "credit_rating": {"fair": 0, "excellent": 1},
#     "buys_computer": {"no": 0, "yes": 1}
# }

# for col, mapping in encoding.items():
#     df[col] = df[col].map(mapping)


# def calculate_entropy(y):
#     class_counts = y.value_counts().to_numpy()
#     total = len(y)
#     entropy_value = -np.sum((count / total) * math.log2(count / total) for count in class_counts)
#     return entropy_value

# def calculate_weighted_entropy(df, attribute, target):
#     total = len(df)
#     weighted_entropy_value = 0

#     for value in df[attribute].unique():
#         subset = df[df[attribute] == value]
#         subset_entropy = calculate_entropy(subset[target])
#         weighted_entropy_value += (len(subset) / total) * subset_entropy
#     return weighted_entropy_value

# def information_gain(df, attribute, target):
#     total_entropy = calculate_entropy(df[target])
#     attribute_entropy = calculate_weighted_entropy(df, attribute, target)
#     return total_entropy - attribute_entropy

# def find_best_split(df, attributes, target):
#     info_gains = {attr: information_gain(df, attr, target) for attr in attributes}
#     return max(info_gains, key=info_gains.get)

# class TreeNode:
#     def __init__(self, attribute=None, value=None, is_leaf=False, prediction=None):
#         self.attribute = attribute
#         self.value = value
#         self.is_leaf = is_leaf
#         self.prediction = prediction
#         self.children = []

# def build_tree(df, attributes, target):
#     """Recursively build the decision tree."""
#     if len(df[target].unique()) == 1:
#         return TreeNode(is_leaf=True, prediction=df[target].iloc[0])
#     if not attributes:
#         return TreeNode(is_leaf=True, prediction=df[target].mode()[0])
#     best_attr = find_best_split(df, attributes, target)
#     root = TreeNode(attribute=best_attr)
#     for value in df[best_attr].unique():
#         subset = df[df[best_attr] == value]
#         if subset.empty:
#             root.children.append(TreeNode(is_leaf=True, prediction=df[target].mode()[0]))
#         else:
#             new_attributes = [attr for attr in attributes if attr != best_attr]
#             child_node = build_tree(subset, new_attributes, target)
#             child_node.value = value
#             root.children.append(child_node)
#     return root

# def print_tree(node, depth=0):
#     indent = "  " * depth
#     if node.is_leaf:
#         print(f"{indent}Predict: {node.prediction}")
#     else:
#         print(f"{indent}{node.attribute}")
#         for child in node.children:
#             print(f"{indent}  {node.attribute} = {child.value}")
#             print_tree(child, depth + 1)

# attributes = ['age', 'income', 'student', 'credit_rating']
# target = 'buys_computer'
# decision_tree = build_tree(df, attributes, target)
# print_tree(decision_tree)


# 8-) decision tree (library)
# import pandas as pd
# file_path=r"D:\Dilshad\DUK\MI\Lab_Assig\lab files\decision tree.csv"
# dataset=pd.read_csv(file_path)
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# for column in dataset.columns:
#     dataset[column] = label_encoder.fit_transform(dataset[column])
# x=dataset.drop('buys_computer',axis=1)
# y=dataset['buys_computer']
# from sklearn.tree import DecisionTreeClassifier
# model=DecisionTreeClassifier()
# model.fit(x,y)
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 8))
# plot_tree(model, filled=True, feature_names=x.columns.tolist(), class_names=['No', 'Yes'])
# plt.show()


#navie bais
import re
from collections import defaultdict
import math

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()  # Tokenize into words

# Function to train Naive Bayes Classifier
def train_naive_bayes(data):
    spam_count = 0
    ham_count = 0
    word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
    total_words = {'spam': 0, 'ham': 0}

    for label, message in data:
        words = preprocess_text(message)
        if label == 'spam':
            spam_count += 1
            for word in words:
                word_counts['spam'][word] += 1
                total_words['spam'] += 1
        else:
            ham_count += 1
            for word in words:
                word_counts['ham'][word] += 1
                total_words['ham'] += 1

    prior_spam = spam_count / len(data)
    prior_ham = ham_count / len(data)
    vocabulary = set(word_counts['spam'].keys()).union(set(word_counts['ham'].keys()))

    return {
        'prior_spam': prior_spam,
        'prior_ham': prior_ham,
        'word_counts': word_counts,
        'total_words': total_words,
        'vocabulary': vocabulary,
    }

# Function to predict if a message is spam or ham
def predict_naive_bayes(model, message):
    words = preprocess_text(message)
    spam_score = math.log(model['prior_spam'])
    ham_score = math.log(model['prior_ham'])

    for word in words:
        # Calculate probabilities with Laplace Smoothing
        spam_word_prob = (model['word_counts']['spam'][word] + 1) / (model['total_words']['spam'] + len(model['vocabulary']))
        ham_word_prob = (model['word_counts']['ham'][word] + 1) / (model['total_words']['ham'] + len(model['vocabulary']))
        spam_score += math.log(spam_word_prob)
        ham_score += math.log(ham_word_prob)

    return 'spam' if spam_score > ham_score else 'ham'

# Example Dataset
data = [
    ('spam', "Win a free prize now"),
    ('ham', "Are you coming to the meeting?"),
    ('spam', "Limited offer, click to claim your reward"),
    ('ham', "Hello, how have you been?"),
    ('spam', "Exclusive deal just for you"),
    ('ham', "Don't forget the project deadline"),
]

# Train the model
model = train_naive_bayes(data)

# Test the model
test_message = "Win a reward now"
print(f"Message: '{test_message}' is classified as: {predict_naive_bayes(model, test_message)}")
