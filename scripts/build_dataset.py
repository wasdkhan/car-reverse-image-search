import argparse
import random
import os

# --data_dir argument specifies where VMMRdb folder is located
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='VMMRdb', help="Directory with the VMMRdb dataset")
parser.add_argument('--output_dir', default='dataset_split', help="Directory to output text files")

if __name__ == '__main__':
	args = parser.parse_args()
	# make sure dataset directory exists and output dir does not exist
	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
	
	# get list of categories (name of vehicles) and sort abc order
	categories = os.listdir(args.data_dir)
	categories.sort()
	# dict to store category name and number of images inside
	category_size_dict = {}
	# overall filenames list
	train_filenames = []
	dev_filenames = []
	test_filenames = []
	for category in categories:
		# get directory name and list of images inside
		category_dir = os.path.join(args.data_dir, category)
		category_filenames = os.listdir(category_dir)

		# get number of images inside and store in dict
		category_size = len(category_filenames)
		category_size_dict[category] = category_size

		# sort filenames abc order and join to create path
		category_filenames.sort()
		category_filenames = [os.path.join(category, f) for f in category_filenames]
			
		# sorted above and shuffled with the same random seed 230 to provide the same shuffle everytime 
		random.seed(230)
		random.shuffle(category_filenames)

		# split into 80% train, 10% val, and 10% test filenames
		# if only 1 goes into test set, if 2 goes into train and test
		split_1 = int(0.8 * category_size)
		split_2 = int(0.9 * category_size)
		category_train_filenames = category_filenames[:split_1]
		category_dev_filenames = category_filenames[split_1:split_2]
		category_test_filenames = category_filenames[split_2:]

		# add onto the overall filenames list
		train_filenames.extend(category_train_filenames)
		dev_filenames.extend(category_dev_filenames)
		test_filenames.extend(category_test_filenames)

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	else:
		print("Warning: output dir {} already exists.".format(args.output_dir))

	# write lists to file
	with open(os.path.join(args.output_dir, "train.txt"), "w") as f:
		for train_filename in train_filenames:
			f.write("{}\n".format(train_filename))

	with open(os.path.join(args.output_dir, "dev.txt"), "w") as f:
		for dev_filename in dev_filenames:
			f.write("{}\n".format(dev_filename))

	with open(os.path.join(args.output_dir, "test.txt"), "w") as f:
		for test_filename in test_filenames:
			f.write("{}\n".format(test_filename))

	# write category_sizes to file
	with open(os.path.join(args.output_dir, "category_sizes.txt"), "w") as f:
		for category in categories:
			f.write("{} {}\n".format(category, category_size_dict[category]))


