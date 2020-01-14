import faiss
import numpy as np

if __name__ == '__main__':
	root_dir = 'dataset_split'
	features_dir = 'dataset_split_features'
	image_dir = 'VMMRdb'

	train_features_filename = 'train_features.npy'
	train_labels_filename = 'train.txt'

	dev_features_filename = 'dev_features.npy'
	dev_labels_filename = 'dev.txt'

	test_features_filename = 'test_features.npy'
	test_labels_filename = 'test.txt'

	train_features = np.load("{}/{}".format(features_dir, train_features_filename))
	# should be (n, 2048), where n is the number of vectors
	(n, dimension) = train_features.shape
	train_labels = []
	with open("{}/{}".format(root_dir, train_labels_filename), 'r') as f:
		train_labels = [line.strip() for line in f]

	dev_features = np.load("{}/{}".format(features_dir, dev_features_filename))
	dev_labels = []
	with open("{}/{}".format(root_dir, dev_labels_filename), 'r') as f:
		dev_labels = [line.strip() for line in f]

	test_features = np.load("{}/{}".format(features_dir, test_features_filename))
	test_labels = []
	with open("{}/{}".format(root_dir, test_labels_filename), 'r') as f:
		test_labels = [line.strip() for line in f]


	k = 5
	input_features = dev_features
	input_labels = dev_labels


	#index = faiss.IndexFlatL2(dimension)
	#print(index.is_trained)
	#index.add(train_features)
	#print(index.ntotal)

	#nlist = 4096
	#quantizer = faiss.IndexFlatL2(dimension)
	#index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
	#assert not index.is_trained
	#index.train(train_features)
	#assert index.is_trained
	#index.add(train_features)
	#print(index.ntotal)

	index = faiss.IndexHNSWFlat(dimension, 32)
	index.verbose = True
	print(index.is_trained)
	index.add(train_features)
	print(index.ntotal)
	index.hnsw.efSearch = 16

	D, I = index.search(input_features, k)
	
	num_input_features = len(input_features)
	top_1_correct = 0
	top_k_correct = 0

	for i in xrange(num_input_features):
		input_label = input_labels[i].split('/')[0]
		print("input label", i, input_label)
		suggested_labels = []
		for idx in I[i]:
			suggested_label = train_labels[idx].split('/')[0]
			suggested_labels.append(suggested_label)
		print("suggested_labels", suggested_labels)
		for i in xrange(len(suggested_labels)):
			suggested_label = suggested_labels[i]
			if i == 0 and suggested_label == input_label:
				top_1_correct += 1
				top_k_correct += 1
				break
			elif suggested_label == input_label:
				top_k_correct += 1
				break

	top_1_accuracy = top_1_correct / float(num_input_features)
	top_k_accuracy = top_k_correct / float(num_input_features)

	print("top 1 accuracy:", top_1_accuracy)
	print("top {} accuracy:".format(k), top_k_accuracy)

	# random_idx = np.random.randint(n)
	# random_feature = np.expand_dims(features[random_idx], axis=0)
	# random_feature_label = labels[random_idx]
	
	# print("input label", random_idx, random_feature_label)

	# k = 5
	# D, I = index.search(random_feature, k)
	# print(I)
	# print(D)

	# for idx in I[0]:
	# 	similar_label = labels[idx]
	# 	print("similar label", idx, similar_label)
	
