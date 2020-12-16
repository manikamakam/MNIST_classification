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



def svm_model(X_train, y_train, X_test, y_test, kernel,transform,C, gamma):

	if C == 'default' and gamma == 'default':
		model = SVC(kernel=kernel)
	else:
		model = SVC(C = C,gamma= gamma, kernel=kernel)

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

	X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

	fig, axis = plt.subplots(4, 4, figsize=(12, 14))
	for i, ax in enumerate(axis.flat):
	    ax.imshow(X_test__[i], cmap='binary')
	    ax.set(title = f"Real Number is {y_test[i]}\nPredicted Number is {y_pred[i]}");

	plt.savefig(f'transform={transform}_kernel={kernel}_C={C}_gamma={gamma}_results.png')
	plt.close()


def hyperparameter_tuning(X_train, y_train, X_test, y_test, kernel):
	# creating a KFold object with 5 splits 
	folds = KFold(n_splits = 5, shuffle = True, random_state = 10)

	# specify range of hyperparameters
	# Set the parameters by cross-validation
	hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
	                     'C': [5,10]}]


	# specify model
	model = SVC(kernel=kernel)

	# set up GridSearchCV()
	model_cv = GridSearchCV(estimator = model, 
	                        param_grid = hyper_params, 
	                        scoring= 'accuracy', 
	                        cv = folds, 
	                        verbose = 1,
	                        return_train_score=True)      

	# fit the model
	print("hereeeeee")
	model_cv.fit(X_train, y_train)
	print("doneeee")

	cv_results = pd.DataFrame(model_cv.cv_results_)

	# converting C to numeric type for plotting on x-axis
	cv_results['param_C'] = cv_results['param_C'].astype('int')

	# # plotting
	plt.figure(figsize=(16,8))

	# subplot 1/3
	plt.subplot(131)
	gamma_01 = cv_results[cv_results['param_gamma']==0.01]

	plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
	plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title("Gamma=0.01")
	plt.ylim([0.60, 1])
	plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
	plt.xscale('log')

	# subplot 2/3
	plt.subplot(132)
	gamma_001 = cv_results[cv_results['param_gamma']==0.001]

	plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
	plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title("Gamma=0.001")
	plt.ylim([0.60, 1])
	plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
	plt.xscale('log')


	# subplot 3/3
	plt.subplot(133)
	gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

	plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
	plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title("Gamma=0.0001")
	plt.ylim([0.60, 1])
	plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
	plt.xscale('log')

	# printing the optimal accuracy score and hyperparameters
	best_score = model_cv.best_score_
	best_hyperparams = model_cv.best_params_

	print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))



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

	print(X_train.shape)
	print(X_test.shape)

	# X_train = scale(X_train)
	# X_test = scale(X_test)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	if args.transform == 'PCA':
		n_components = 16
		pca = PCA(n_components=n_components, svd_solver='randomized',
		          whiten=True)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		plt.hist(pca.explained_variance_ratio_, bins=n_components, log=True)
		plt.savefig(f'transform=PCA_kernel={args.kernel}_hist.png')
		print("Explained Variance ration:", pca.explained_variance_ratio_.sum())
		plt.close()

	if args.transform == 'LDA':
		lda = LDA(n_components=16)
		X_train = lda.fit_transform(X_train, y_train)
		X_test = lda.transform(X_test)




	C= 'default'
	gamma = 'default'

	# svm_model(X_train, y_train, X_test, y_test, args.kernel,args.transform, C, gamma)
	hyperparameter_tuning(X_train, y_train, X_test, y_test, args.kernel)

