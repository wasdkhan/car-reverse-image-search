# flask_app/settings.py

# filenames of feature vectors and labels
train_features_filename = 'train_features.npy'
train_labels_filename = 'train.txt'
index_filename = 'train_vector.index'

# train labels are in text file with each line representing the corresponding label to the feature vector
train_labels = []
with open(train_labels_filename, 'r') as f:
	train_labels = [line.strip() for line in f]

labels = train_labels

PORT = 8080
