
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn import svm

def get_fake_data():

	#num_examples x num tactile sensors
	X = np.zeros((1000, 96), dtype=np.float32)

	#stable or not. Just look force closure, so volume > 0
	Y = np.random.random_integers(low=0,high=1, size=(1000, 1))

	train_X = X[0:800]
	test_X = X[800:900]

	train_Y = Y[0:800]
	test_Y = Y[800:900]

	return (train_X, train_Y, test_X, test_Y)

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# def get_adjustment_data():
# 	"""
# 	follow the function get_stability_classification_data
# 	but iterate over the other collection of pertubations
# 	with has f0, f1, .. or grasp0, grasp1
# 	X will be exactly the same as get_stability_classification_data 
# 	but the Y will be the sign of the difference in the Volume between
# 	f0 and f1 
# 	and only keep X and Y for a all the examples with a single perturbation type.
# 		"transf" : {
# 		"translation" : {
# 			"0" : 0,
# 			"1" : 0,
# 			"2" : 0
# 		},
# 		"orientation" : {
# 			"0" : 0.9921976672293292,
# 			"1" : 0,
# 			"2" : -0.12467473338522772,
# 			"3" : 0
# 		}
# 	} 
# 	one for each of 
# 	perturbations.push_back(rotXYZ(0.25, 0, 0));
#     perturbations.push_back(rotXYZ(0, 0.25, 0));
#     perturbations.push_back(rotXYZ(0, 0, 0.25));
#     perturbations.push_back(rotXYZ(-0.25, 0, 0));
#     perturbations.push_back(rotXYZ(0, -0.25, 0));
#     perturbations.push_back(rotXYZ(0, 0, -0.25));
#     perturbations.push_back(translate_transf(vec3(0,0, -20)));
#     perturbations.push_back(translate_transf(vec3(0,0, 20)));

# 	"""
# 	break
def get_adjustment_data():
	from pymongo import MongoClient

	mongo_url = os.getenv("MONGO_URL")
	client = MongoClient(mongo_url)
	db = client.get_default_database()
	cursor = db.perturbations.find()

	num_frames = cursor.count()

	#num_examples x num tactile sensors
	X = np.zeros((num_frames, 96), dtype=np.float32)
	#stable or not. Just look force closure, so volume > 0
	Y = np.zeros((num_frames, ), dtype=np.float32)
        transfs = []
	transf  = None
	count = 0
	for perturbation_frame in cursor:

		qw = perturbation_frame['transf']['orientation']['0']
		qx = perturbation_frame['transf']['orientation']['1']
		qy = perturbation_frame['transf']['orientation']['2']
		qz = perturbation_frame['transf']['orientation']['3']
		x = perturbation_frame['transf']['translation']['0']
		y = perturbation_frame['transf']['translation']['1']
		z = perturbation_frame['transf']['translation']['2']

		transf_temp = tuple([qw, qx, qy, qz, x, y, z])


		if transf is None:
			transf = transf_temp

		if transf != transf_temp:
			print "skipping this transf becaue it is not the transformation we want"
			continue


		oid = perturbation_frame["f0"]
		grasp_frame_0 = db.grasps.find({"_id": oid}).next()

		for j, tactile in enumerate(grasp_frame_0["tactile"]):
			X[count, j] = tactile["force"]
		volume0 = grasp_frame_0['grasp']['energy']['Volume']

		oid = perturbation_frame["f1"]
		grasp_frame_1 = db.grasps.find({"_id": oid}).next()
		volume1 = grasp_frame_1['grasp']['energy']['Volume']

		volume_delta = volume1 - volume0
		Y[count] = volume_delta

		count += 1 

	X = X[:count]
	Y = Y[:count]

	Y = np.sign(Y)

	X,Y = shuffle_in_unison_inplace(X,Y)


	positive_count = len(Y[Y > 0])
	print "X.shape: " + str(X.shape)
	print "positive count: " + str(positive_count)
	pos_percent =  positive_count / (1.0* X.shape[0])
	pos_percent = max(pos_percent, 1-pos_percent)
	print "always guess positive: " + str( pos_percent)

	train_X = X[:int(count*0.8)]
	train_Y = Y[:int(count*0.8)]
	test_X = X[int(count*0.8):]
	test_Y = Y[int(count*0.8):]
	
	return (train_X, train_Y, test_X, test_Y)



def get_stability_classification_data():
	from pymongo import MongoClient

	mongo_url = os.getenv("MONGO_URL")
	client = MongoClient(mongo_url)
	db = client.get_default_database()
	cursor = db.grasps.find()
	grasps_count = cursor.count()
	# import IPython
	# IPython.embed()
	#num_examples x num tactile sensors
	X = np.zeros((grasps_count, 96), dtype=np.float32)
	#stable or not. Just look force closure, so volume > 0
	Y = np.zeros((grasps_count, ), dtype=np.float32)

	count = 0
	for  doc in cursor:
		for j, tactile in enumerate(doc["tactile"]):
			X[count, j] = tactile["force"]
		Y[count] = doc["grasp"]["energy"]["Volume"]

		# if at least one tactile sensor is active:
		if(X[count].max() != 0):
			count += 1

	print "Removing entries where all tactile contacts are 0"
	X = X[:count]
	Y = Y[:count]

	Y[Y <= 8] = -1
	Y[Y > 8] = 1


	X,Y = shuffle_in_unison_inplace(X,Y)

	#try projecting data into 10 dimension space rather than 96
	#X = PCA(X)
	#print X.shape
	#should be (num_examples, 10) now not (num_examples, 96)

	train_X = X[:int(count*0.8)]
	train_Y = Y[:int(count*0.8)]
	test_X = X[int(count*0.8):]
	test_Y = Y[int(count*0.8):]
	
	return (train_X, train_Y, test_X, test_Y)

#this will take 
#X 10K x 96
#
# def train_adjustment_classifier():
# 	break

def train_logistic_regression_classifier(train_X, train_Y, test_X, test_Y):

	print "train_X.shape" + str(train_X.shape)

	regressor = LogisticRegression(
		penalty='l2',
		dual=False,
		tol=0.00001, 
		C=1.0, 
		fit_intercept=True, 
		intercept_scaling=1, 
		class_weight="balanced", 
		random_state=None, 
		solver='liblinear', 
		max_iter=1000, 
		multi_class='ovr', 
		verbose=1, 
		warm_start=False, 
		n_jobs=1)

	regressor.fit(train_X, train_Y)

	train_score = regressor.score(train_X, train_Y)
	test_score = regressor.score(test_X, test_Y)

	#compute score for guessing largest category every time
	positive_train_count = len(train_Y[train_Y == 1])
	zero_train_count = len(train_Y[train_Y == 0])
	negative_train_count = len(train_Y[train_Y == -1])

	largest_count = max([positive_train_count, zero_train_count, negative_train_count])
	total = sum([positive_train_count, zero_train_count, negative_train_count])
	largest_category_percent =  (largest_count * 1.0) / total

	print "percentage of training data in largest single category: " + str(largest_category_percent)
	print "regressor score for train data:" + str(train_score)
	print "regressor score for test data:" + str(test_score)

def train_svm_classifier(train_X, train_Y, test_X, test_Y):

	print "train_X.shape" + str(train_X.shape)

		
            
        clf = svm.SVC(gamma=0.001, C=100)
        clf.fit(train_X, train_Y)

	#import IPython
	#IPython.embed()
	train_score = clf.score(train_X, train_Y)
	test_score = clf.score(test_X, test_Y)

	#compute score for guessing largest category every time
	positive_train_count = len(train_Y[train_Y == 1])
	zero_train_count = len(train_Y[train_Y == 0])
	negative_train_count = len(train_Y[train_Y == -1])

	largest_count = max([positive_train_count, zero_train_count, negative_train_count])
	total = sum([positive_train_count, zero_train_count, negative_train_count])
	largest_category_percent =  (largest_count * 1.0) / total

	print "percentage of training data in largest single category: " + str(largest_category_percent)
	print "regressor score for train data:" + str(train_score)
	print "regressor score for test data:" + str(test_score)


def pca(X, feature_count):
    pca = PCA(n_components=feature_count)
    pca.fit(X)
    X = pca.transform(X)
    return X


def pca_train_logistic_regression_classifier(new_train_X, train_Y, new_test_X, test_Y, feature_count):

    train_logistic_regression_classifier(pca(train_X, feature_count), train_Y, pca(test_X, feature_count), test_Y)
    
    return

def pca_train_classifier(algo_func, feature_count, new_train_X, train_Y, new_test_X, test_Y):

    algo_func(pca(train_X, feature_count), train_Y, pca(test_X, feature_count), test_Y)
    return

if __name__ == "__main__":
	#get data 

	#get fake data:
	#train_X, train_Y, test_X, test_Y = get_fake_data()
	#use this is you want to classifiy stability.
	train_X, train_Y, test_X, test_Y = get_stability_classification_data()
        
	#use this is you want to run with adjustment data
	#train_X, train_Y, test_X, test_Y = get_adjustment_data()

	#train classifier
	# train_logistic_regression_classifier(train_X, train_Y, test_X, test_Y)
        for count in [96, 30, 20, 10]:
            print("\nFeature Count: " + str(count))
            print("logistic_regression:\n")
	    pca_train_classifier(train_logistic_regression_classifier, count, train_X, train_Y, test_X, test_Y)
            print("svm:\n")
	    pca_train_classifier(train_svm_classifier, count, train_X, train_Y, test_X, test_Y)
	#useful commands:

	#if you want to plot a histogram:
	#import matplotlib.pyplot as plt
	#plt.hist(train_Y)
	#plt.show()









