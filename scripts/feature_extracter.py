import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np

import resnet_50

txt_filename = 'dataset_split/test.txt'

class TextFileDataset(Dataset):
	"""Text File dataset."""

	def __init__(self, txt_file, root_dir, transform=None):
		"""
		Args:
		    txt_file (string): Path to the txt file.
		    root_dir (string): Directory with all the images.
		    transform (callable, optional): Optional transform to be applied
			on a sample.
		"""
		self.image_filenames = []
		with open(txt_file, "r") as f:
			self.image_filenames = [line.strip() for line in f]
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.image_filenames)

	def __getitem__(self, idx):
		image_path = "{}/{}".format(self.root_dir, self.image_filenames[idx])
		# important to open as RGB because images maybe grayscale (1 channel)
		image = Image.open(image_path).convert('RGB')
		if self.transform:
			image = self.transform(image)

		return image
	
if __name__ == '__main__':
	# load cpu and gpu device if available
	cpu_device = torch.device('cpu')
	device = cpu_device
	if torch.cuda.is_available():
		device = torch.device("cuda:0")

	# load trained cars resnet model
	model = resnet_50.resnet_50
	model.load_state_dict(torch.load('resnet_50.pth'))

	# remove last layer (FC) and do not require gradient, set to eval mode
	modules = list(model.children())[:-1]
	model = nn.Sequential(*modules)
	for p in model.parameters():
		p.requires_grad = False
	model.to(device)
	model.eval()

	# pre-processing transforms
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize
	])

	# load list of images from text file and use pre-processing transforms
	textfile_dataset = TextFileDataset(txt_file=txt_filename, 
					root_dir='VMMRdb', 
					transform=preprocess)

	# load data by batch size of 10 with 4 workers and do not randomly shuffle
	dataloader = DataLoader(textfile_dataset, batch_size=10, shuffle=False, num_workers=4)

	# iterate through each batch and run inference on GPU (device), concat onto CPU tensor every 1000 batches 
	outputs = []
	final_tensor = torch.Tensor().to(cpu_device)
	for i, image_batch in enumerate(dataloader):
		print(i, image_batch.shape)
		image_batch = image_batch.to(device)
		output = model(image_batch)
		outputs.append(output)
		print(i, output.shape)
		if i > 1 and i % 1000 == 0:
			output_tensor = torch.cat(outputs).to(cpu_device)
			final_tensor = torch.cat((final_tensor, output_tensor))
			outputs = []

	# remaining tensors under 
	output_tensor = torch.cat(outputs).to(cpu_device)
	final_tensor = torch.cat((final_tensor, output_tensor))
	outputs = []

	# save CPU tensor as numpy array to final_tensor.npy file
	# torch.save(final_tensor, 'final_tensor.pt')
	np.save('final_tensor.npy', final_tensor.data.numpy())

	# image_filename = "sample_car_images/toyota-rav4-2011.jpg"
	# im = Image.open(image_filename)
	# img_tensor = preprocess(im)

	# input_tensor = torch.Tensor(2, 3, 224, 224)
	# input_tensor[0] = img_tensor
	# input_tensor[1] = img_tensor
	# input_tensor_var = Variable(input_tensor)
	# features_var = model(input_tensor_var)
	# features = features_var.data

