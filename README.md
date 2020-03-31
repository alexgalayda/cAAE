# Быстрый старт
Просто запустите `run.sh` вместе с именем файла конфигурации из папки с конфигами (config/{file}.env)  
Напримаер: ```$ run.sh test```

# cAAE

code for Unsupervised Detection of Lesions in Brain MRI using constrained adversarial auto-encoders, [https://arxiv.org/abs/1806.04972](https://arxiv.org/abs/1806.04972)

AAE model is implemented based on [this github repo](https://github.com/Naresh1318/Adversarial_Autoencoder)

Required: python>=3.6, tensorlayer, tensorflow>=1.0, numpy

train model:  python main.py --model_name "cAAE" --z_dim "128"

# Using:
First, you need to access an open dataset (I took part of the code from [here](https://github.com/jokedurnez/HCP_download)):
1. You'll need to create HCP credentials. You'll need to accept the terms of data usage as well. You can do so by following [this tutorial](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS).
Stop at this moment and use the received keys
![keys](https://wiki.humanconnectome.org/download/attachments/67666030/image2015-1-7%2014%3A41%3A22.png?version=1&modificationDate=1420664134386&api=v2)

2. Use the data instead of XXX in the file `./doc_scripts/get_dataset/credentials`:
```
[hcp]
AWS_ACCESS_KEY_ID=XXX
AWS_SECRET_ACCES_KEY=XXX
```
3. Now let's get started
```
./run.sh -m /path/to/HCP
```
You can use  
* -m set the directory where the dataset will be downloaded (by default /mnt/storage/datasets/HCP/)
* -g attribute to set the number of the video card (ALL for all video cards)
* -s to specify the folder for sharing the folder (there will be a folder in docker /root/shara)
```
./run.sh -g 0 -s /path/to/shara -m /path/to/HCP
```
