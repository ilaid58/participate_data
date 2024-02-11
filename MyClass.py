import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier


class getValue:
	def getValueEdition(self, edition):
		list_data = []
		data = ''
		for i in edition:
			for j in i:
				if j == ',':
					break
				else:
					data = data+j
			list_data.append(data)
			data = ''
		edition.update(list_data)

	
	def getValueReview(self, review):
		list_data = []
		data = ''
		for i in review:
				for j in i:
					if j == ' ':
						break
					else:
						data = data+j
				list_data.append(float(i[0])/5)	

		review.update(list_data)
		review = pd.to_numeric(review)
		return review
	
	def getValueRating(self, rating):
		list_data = []
		for i in rating:
			list_data.append(i[0])
		rating.update(list_data)
		rating = pd.to_numeric(rating)
		return rating
			

class read_file:
	def __init__(self, data_file):
		self.data_file = data_file
		
	def read_csv(self):
		self.data_file = self.data_file.split(', ')
		for i in self.data_file:
			df = pd.read_csv(i)
			yield df
	
	def read_excel(self):
		self.data_file = self.data_file.split(', ')
		for i in self.data_file:
			df = pd.read_excel(i)
			yield df

class object_to_npArray:
	def __init__(self, data, feature):
		self.data = data
		self.feature = feature
	
	def convert(self):
		
		data_list = []
		for i in self.data[self.feature]:
			data = i.split(' ')
			data_list.append(data)
		data_list = np.array(data_list)
		self.data[self.feature].update(data_list)
		
		

class data_to_num:
	def __init__(self, data, feature):
		self.data = data
		self.feature = feature

	def convert(self):

		data_groupby = self.data.groupby(self.feature)[self.feature]
		
		final_data = []
		self.data_name_num = []
		data_num = []
		data_name = []
		cpt = 0
		
		for i in data_groupby:
			cpt = cpt+1
			self.data_name_num.append([i[0], cpt])
			data_name.append(i[0])
			data_num.append(cpt)
				
		for i in self.data[self.feature]:
			for j in self.data_name_num:
				if i == j[0]:
					self.data[self.feature] = self.data[self.feature].replace(i, j[1])
		
		self.data[self.feature] = self.data[self.feature].fillna(0)

		#self.data['Age'] = self.data['Age'].fillna(0)
		#self.data['Fare'] = self.data['Fare'].fillna(0)


class feature_target:
	def __init__(self, df, feature, target):
		self.df = df
		self.feature = feature
		self.target = target
	
	def feature(self):
		self.x = self.df[self.feature]
		return self.x

	def target(self):
		self.y = self.df[self.target]
		return self.y


class linear_regression:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self):
		model = LinearRegression()
		model.fit(self.x_train, self.y_train)
		y_pred = model.predict(self.x_test)
		return y_pred


class logistic_regression:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self):
		model = LogisticRegression()
		model.fit(self.x_train, self.y_train)
		y_pred = model.predict(self.x_test)
		
		return y_pred


class kfold:
	def __init__(self, n_splits, X, y, random):
		self.n_splits = n_splits
		self.X = X
		self.y = y
		self.random = random
	
	def splits(self):
		kf = KFold(n_splits = self.n_splits, shuffle = True, random_state = self.random)
		splits = list(kf.split(self.X, self.y))
		train_test = []
		for i in splits:
			train, test = i
			x_train = self.X.iloc[train]
			x_test = self.X.iloc[test]
			
			y_train = self.y.iloc[train]
			y_test = self.y.iloc[test]
			
			train_test.append([x_train, x_test, y_train, y_test])
			
		return train_test


class decision_tree_classifier:
	def __init__(self, x_train, x_test, y_train):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train

	def prediction(self):
		model = DecisionTreeClassifier()
		model.fit(self.x_train, self.y_train)
		y_pred = model.predict(x_test)
		
		return y_pred
	
	def gridSearch(self, max_depth, min_sample_leaf, max_leaf_node, cv):
		param_grid ={	'max_depth':max_depth,
						'min_samples_leaf':min_sample_leaf,
					    'max_leaf_nodes': max_leaf_node
					}
		
		dt = DecisionTreeClassifier()
		gs = GridSearchCV(dt, param_grid, scoring = 'accuracy', cv = cv)
		print(gs.estimator.get_params().keys())
		gs.fit(self.x_train, self.y_train)

		pred = gs.predict(self.x_test)
		print('best parameter : ',gs.best_params_)
		best_score = gs.best_score_
		print('best score : ',best_score)
		
		return pred 


class random_forest:
	def __init__(self, x_train, x_test, y_train):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		
	def prediction(self):
		rf = RandomForestClassifier()
		rf.fit(self.x_train, self.y_train)
		pred = rf.predict(x_test)
		
		return [pred, best_score]
	
	def gridSearch(self, n_estimators, cross_valid):
		param_grid = {
						'n_estimators':n_estimators
						}
		rf = RandomForestClassifier()
		gs = GridSearchCV(rf, param_grid, scoring = 'accuracy', cv = cross_valid)
		gs.fit(self.x_train, self.y_train)

		pred = gs.predict(self.x_test)
		best_score = gs.best_score_
		print('best score : ',best_score)
		return [pred, best_score]


class neural_network_classifier:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self, max_iter, hidden_layer, alpha, random):
		mlp = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer, alpha=alpha, random_state=random)
		mlp.fit(self.x_train, self.y_train)
		pred = mlp.predict(self.x_test)
		print(mlp.predict_proba(self.x_test))
		return pred
	
	def gridSearch(self, max_iter, hidden_layer, alpha, random, cv):
		mlp = MLPClassifier()
		param = {'max_iter':[500, max_iter], 'hidden_layer_sizes':[hidden_layer,100,150,200], 'alpha':[alpha,0.00011], 'random_state':[20,21,22,27,30]}
		gs = GridSearchCV(mlp, param, scoring = 'accuracy', cv = cv)
		gs.fit(self.x_train, self.y_train)
		pred = gs.predict(self.x_test)
		print(gs.best_score_)
		print(gs.predict_proba(self.x_test))
		return [pred, gs.predict_proba(self.x_test)]


class neural_network_regression:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self, max_iter, hidden_layer, alpha, random):
		mlr = MLPRegressor(max_iter = max_iter, hidden_layer_sizes = hidden_layer, alpha = alpha, random_state = random)
		mlr.fit(self.x_train, self.y_train)
		pred = mlr.predict(self.x_test)
		return pred
	
	def gridSearch(self, param, cv):
		pred = []
		mlr = MLPRegressor()
		grid = GridSearchCV(mlr, param, cv = cv)
		grid.fit(self.x_train, self.y_train)
		pred.append(grid.predict(self.x_test))
		pred.append(grid.predictproba(self.x_test))
		return pred

class neighbor_classifier:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self, n):
		knn = KNeighborsClassifier(n_neighbors=n)
		knn.fit(self.x_train, self.y_train)
		pred = knn.predict(self.x_test)
		print(knn.predict_proba(self.x_test))
		return pred
		
	def gridSearch(self, param, cv):
		gridParam = {'n_neighbors':np.array(param)}
		knn = KNeighborsClassifier()
		grid = GridSearchCV(knn, gridParam, cv=cv)
		grid.fit(self.x_train, self.y_train)
		knn_result = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'])
		knn_result.fit(self.x_train, self.y_train)
		pred = knn_result.predict(self.x_test)
		print(knn_result.predict_proba(self.x_test))
		return pred
	
		
		
class calcul_metrics:
	def __init__(self, y_test, y_pred, index_data):
		self.y_test = y_test
		self.y_pred = y_pred
		self.index_data = index_data
		
	def accuracy(self):
		accuracy = accuracy_score(self.y_test, self.y_pred)
		return [self.index_data, accuracy, self.y_pred]
	
	def f1(self):
		f1 = f1_score(self.y_test, self.y_pred)
		return [self.index_data, f1, self.y_pred]
	
	def precision(self):
		precision = precision_score(self.y_test, self.y_pred)
		return [self.index_data, precision, self.y_pred]
	
	def recall(self):
		recall = recall_score(self.y_test, self.y_pred)
		return [self.index_data, recall, self.y_pred]


class max_metrics:
	def __init__(self, metrics_list):
		self.metrics_list = metrics_list
		
	def find_max(self):
		max_metric = -1
		data_maxMetric = []
		for i in self.metrics_list:
			if max_metric < i[1]:
				max_metric = i[1]
				data = i[0]
				y_pred = i[2]
		data_maxMetric.append([data, max_metric])
		return [data, max_metric, y_pred]
