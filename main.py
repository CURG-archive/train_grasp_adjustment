
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

def get_fake_data():

	#num_examples x num tactile sensors
	X = np.zeros((1000, 96), dtype=np.float32)

	#stable or not. Just look force closure, so volume > 0
	Y = np.random.random_integers(low=0,high=1, size=(1000, 1))

	train_X = X[0:800]
	test_X = X[800:900]
	validate_X = X[900:1000]

	train_Y = Y[0:800]
	test_Y = Y[800:900]
	validate_Y = Y[900:1000]

	return (train_X, train_Y, test_X, test_Y, validate_X, validate_Y)


def get_data():
	from pymongo import MongoClient
        mongo_url = os.getenv("MONGO_URL")
        client = MongoClient(mongo_url)
        db = client.get_default_database()
        cursor = db.grasps.find()
        for doc in cursor:
            print(doc)





if __name__ == "__main__":
        get_data()
	train_X, train_Y, test_X, test_Y, validate_X, validate_Y = get_fake_data()

	regressor = LogisticRegression(
		penalty='l2',
		dual=False,
		tol=0.0001, 
		C=1.0, 
		fit_intercept=True, 
		intercept_scaling=1, 
		class_weight=None, 
		random_state=None, 
		solver='liblinear', 
		max_iter=100, 
		multi_class='ovr', 
		verbose=0, 
		warm_start=False, 
		n_jobs=1)

	import IPython
	# IPython.embed()

	regressor.fit(train_X,train_Y.flatten())

	print regressor.score(test_X, test_Y)




