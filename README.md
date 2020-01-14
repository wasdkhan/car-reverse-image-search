# car-reverse-image-search
Reverse image search to determine car make and model from image.

Blog post accompanying the code, found [here](https://wasdkhan.github.io/2020/01/12/building-car-recognition.html)

## Code Setup
1. Clone this repository:
```
git clone https://github.com/wasdkhan/car-reverse-image-search/
```
2. Download FAISS [indexes](https://drive.google.com/file/d/1Osf2Ems4XNNLHLkKSBy6Y_rYYyDiNBhp/view?usp=sharing) and [vectors](https://drive.google.com/file/d/1BPZMd-mnDiaYhYUSD8DBDudIyAmSoXPJ/view?usp=sharing), and VMMRdb trained ResNet [model](https://drive.google.com/file/d/1FNBXR-t6cD-2Fuli3x6fr733RiqysUdR/view?usp=sharing), 
and place the three files in the repository (i.e. 'car-reverse-image-search/*').
3. Download and Unzip [VMMRdb](http://vmmrdb.cecsresearch.org/Dataset/VMMRdb.zip) and renamed the folder to VMMRdb and place in /static (i.e. 'car-reverse-image-search/static/VMMRdb')


## Build dependencies from Docker Hub
PyTorch, Torchvision, Flask, and Faiss libraries are needed, they are bundled together in this docker image.
Pull the [image](https://hub.docker.com/r/wasd/car-reverse-image-search-docker/) from Docker Hub like so:
```
docker pull wasd/car-reverse-image-search-docker
```

## To Run
1. Run the docker image and mount the repository to the container.
```
docker run --rm -it -u="$(id -u):$(id -g)" -v="/home/user/code/car-reverse-image-search:/workspace" -p 8080:8080 wasd/car-reverse-image-search-docker
```
where '/home/user/code/car-reverse-image-search' is the absolute directory to the repo.

2. Make the flask server python file executable and run.
```
chmod +x server.py
python server.py
```

3. Visit localhost:8080 in a web browser and upload a picture or paste a url. 
If everything is setup correctly, it should return the top 5 closest looking car images with makes and models. 

## Useful Code and Resources

1. VMMRdb for releasing [car image dataset](http://vmmrdb.cecsresearch.org/)
2. Torch to PyTorch conversion [script](https://github.com/clcarwin/convert_torch_to_pytorch)
