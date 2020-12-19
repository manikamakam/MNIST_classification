import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def svm_model(X_train, y_train, X_test, y_test, kernel,transform,C, gamma, X_test_orig):

	if C == 'default' and gamma == 'default':
		model = SVC(kernel=kernel)
	else:
		model = SVC(C = C,gamma= gamma, kernel=kernel)

	# print(X_train.shape)

	model.fit(X_train, y_train)

	# Test Accuracy
	y_pred = model.predict(X_test)
	print("Test Accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

	# Train Accuracy
	y_pred_train = model.predict(X_train)
	print("Train Accuracy:", metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train), "\n")

	# cm
	mat = confusion_matrix(y_test, y_pred) # Confusion matrix

	# metrics
	print(mat, "\n")

	fig = plt.figure(figsize=(10, 10)) # Set Figure
	# Plot Confusion matrix
	sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values');
	# plt.show();
	plt.savefig(f'transform={transform}_kernel={kernel}_C={C}_gamma={gamma}_CM.png')
	plt.close()

	#plot predictions
	# X_test__ = X_test.reshape(X_test.shape[0], int(np.sqrt(X_test.shape[1])),int(np.sqrt(X_test.shape[1])))
	X_test__ = X_test_orig.reshape(X_test_orig.shape[0], 28, 28)

	fig, axis = plt.subplots(4, 4, figsize=(12, 14))
	for i, ax in enumerate(axis.flat):
	    ax.imshow(X_test__[i], cmap='binary')
	    ax.set(title = f"Real Number is {y_test[i]}\nPredicted Number is {y_pred[i]}");

	plt.savefig(f'transform={transform}_kernel={kernel}_C={C}_gamma={gamma}_results.png')
	plt.close()




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--kernel', type=str, default='linear', help='rbf or poly or linear')
	parser.add_argument('--transform', type=str, default='none', help='PCA or LDA or none')
	args = parser.parse_args()

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	print(X_train.shape)
	print(X_test.shape)

	X_train = X_train.reshape((60000,784))
	X_test = X_test.reshape((10000,784))

	X_test_orig = X_test
	# print(X_train.shape)
	# print(X_test.shape)

	# plt.plot(figure = (16,10))
	# g = sns.countplot( y_train, palette = 'icefire')
	# plt.title('Number of digit classes')
	# plt.xlabel('Count')
	# plt.ylabel('Label');
	# plt.show()

	# train_data.label.astype('category').value_counts()

	X_train = X_train/255.0
	X_test = X_test/255.0



	# X_train = scale(X_train)
	# X_test = scale(X_test)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	

	if args.transform == 'PCA':
		n_components = 169
		pca = PCA(n_components=n_components, svd_solver='randomized',
		          whiten=True)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		plt.hist(pca.explained_variance_ratio_, bins=n_components, log=True)
		plt.savefig(f'explained_variance_ratio_hist.png')
		print("Explained Variance ration:", pca.explained_variance_ratio_.sum())
		plt.close()

	if args.transform == 'LDA':
		lda = LDA(n_components=9)
		X_train = lda.fit_transform(X_train, y_train)
		X_test = lda.transform(X_test)

	print(X_train.shape)
	print(X_test.shape)
	# X_test__ = X_test.reshape(X_test.shape[0], int(np.sqrt(X_test.shape[1])),int(np.sqrt(X_test.shape[1])))
	# print(X_test__.shape[1])
	# X_test__ = X_test_orig.reshape(X_test.shape[0], 28, 28)
	# print(X_test__.shape)

	C= 'default'
	gamma = 'default'
	# C=10
	# gamma=0.001

	svm_model(X_train, y_train, X_test, y_test, args.kernel,args.transform, C, gamma, X_test_orig)
