from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plots
import numpy as nump

class Algorithms:
	def Randomforestclassifier(self,c_d,X_train,Y_train,X_test,Y_test):
		RFC = RandomForestClassifier(n_estimators = 150, max_depth = 20)
		RFC.fit(X_train, Y_train)
		predict = RFC.predict(X_test)
		D_result = list(map(lambda k: c_d[k],predict))
		test_c = list(map(lambda j: c_d[j],Y_test))
		cp = list(map(lambda c: 1 if test_c[c] == D_result[c] else 0,range(0,len(test_c))))
		RFCaccuarcy = cp.count(1)/len(cp)
		print("Random Forest Classifier Accuracy: ",RFCaccuarcy)
		return RFCaccuarcy
		
	def Baggingclassifier(self,c_d,X_train,Y_train,X_test,Y_test):
		BC = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy"),n_estimators=10, random_state=0).fit(X_train, Y_train)
		predict = BC.predict(X_test)
		D_result = list(map(lambda k: c_d[k],predict))
		test_c = list(map(lambda j: c_d[j],Y_test))
		cp = list(map(lambda c: 1 if test_c[c] == D_result[c] else 0,range(0,len(test_c))))
		BCaccuarcy = cp.count(1)/len(cp)
		print("Bagging Classifier Accuracy: ",BCaccuarcy)
		return BCaccuarcy
		
	def Gradientboostingclassifier(self,c_d,X_train,Y_train,X_test,Y_test):
		GB = GradientBoostingClassifier(n_estimators=200, random_state=0)
		GB.fit(X_train, Y_train)
		predict = GB.predict(X_test)
		D_result = list(map(lambda k: c_d[k],predict))
		test_c = list(map(lambda j: c_d[j],Y_test))
		cp = list(map(lambda c: 1 if test_c[c] == D_result[c] else 0,range(0,len(test_c))))
		GBaccuarcy = cp.count(1)/len(cp)
		print("Gradient Boosting Classifier: ",GBaccuarcy)
		return GBaccuarcy
		
	def Knearestneighbors(self,X_train,Y_train,X_test,Y_test):
		s = StandardScaler()
		s.fit(X_train)
		X_train_s = s.transform(X_train)
		X_test_s = s.transform(X_test)
		g = { 'n_neighbors': [5, 6, 7, 16], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan'] }
		KNN = GridSearchCV(KNeighborsClassifier(), g, cv=5, n_jobs=-1)
		KNN.fit(X_train_s, Y_train)
		KNNaccuarcy = KNN.score(X_test_s, Y_test)
		print("K-Nearest Neighbors Accuracy: ",KNNaccuarcy)
		return KNNaccuarcy
	
	def Randomforestclassifiergraph(self,c_d,X_train,Y_train,X_test,Y_test):
		l= []
		m = []
		for i in range(50,311,20):
			RFC = RandomForestClassifier(n_estimators = i)
			RFC.fit(X_train, Y_train)
			predict = RFC.predict(X_test)
			D_result = list(map(lambda k: c_d[k],predict))
			test_c = list(map(lambda j: c_d[j],Y_test))
			cp = list(map(lambda c: 1 if test_c[c] == D_result[c] else 0,range(0,len(test_c))))
			accuarcy = cp.count(1)/len(cp)
			l.append(i)
			m.append(accuarcy)
		plots.plot(l,m)
		plots.xlabel('Tress')
		plots.ylabel('Accuracy')
	
	def Knearestneighborsgraph(self,X_train,Y_train,X_test,Y_test):
		e_rate = []
		for i in range(1,40):
			KNN = KNeighborsClassifier(n_neighbors=i)
			KNN.fit(X_train,Y_train)
			d = KNN.predict(X_test)
			e_rate.append(nump.mean(d != Y_test))
		plots.figure(figsize=(10,6))
		plots.plot(range(1,40),e_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
		plots.xlabel('K Values')
		plots.ylabel('Error Rate')