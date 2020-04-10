# Repeat of the cAAE-experiment
This is a repetition of the experiment from this [article](https://arxiv.org/pdf/1806.04972.pdf) 
entitled "Unsupervised Detection of Lesions in Brain MRI using constrained adversarial auto-encoders". 
The creators posted their code on [this github repo](https://github.com/aubreychen9012/cAAE).
Since the code provided by the researchers raises questions, I want to re-arrange the experiment.

### Project Status:
The original project has a number of problems:
* `wd = "./Data/CamCAN_unbiased/CamCAN/T2w"`: CamCAN - это закрытый датасет
* model.py - не рабочий код

I have already done:
* [tensorflow](https://www.tensorflow.org/) => [pytorch](https://pytorch.org/)
* [nibabel](https://nipy.org/nibabel/) => [antspy](https://github.com/ANTsX/ANTsPy)
* automatic download of [HCP](http://www.humanconnectomeproject.org/data/)
* [tensorboard](https://www.tensorflow.org/tensorboard)
* docker container
* [AAE](https://github.com/eriklindernoren/PyTorch-GAN)

I have to do:
* автоматическое скачивание [BRATS](https://www.med.upenn.edu/sbia/brats2018/data.html) (I was not given access)
* cAAE
* saving weights
* unit test
* CI
* Test jupyter notebook

I'll do it if time is left:
* Two-dimensional convolution is used instead of three-dimensional
* Add Documentation Site on Jekyll


### Quick start

Just run `run.sh` along with the name of the configuration file from the config folder (config/{file}.env)
For example: `$ run.sh example`

If you are not running a jupyter notebook server, then you need to ssh port:  
    ```
    ssh -N -L localhost:6969:localhost:6969 -i ~/work/ANGan/id_rsa_10 galayda@10.55.229.114
    ```

In order to watch training statistics you need a tensorboard:  
    ```
    tensorboard --logdir ./log --port 6006
    ```



### Using:
#### To start training
1. You need to access an open dataset (I took part of the code from [here](https://github.com/jokedurnez/HCP_download)). You'll need to create HCP credentials. You'll need to accept the terms of data usage as well. You can do so by following [this tutorial](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS).
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
    $ run.sh train
    ```
#### To start testing:
We will test on the [BRATS](https://www.med.upenn.edu/sbia/brats2018/data.html) dataset. 
To access the dataset, follow [the instructions](https://www.med.upenn.edu/sbia/brats2018/registration.html) on the official website.
At the end you should have a `BRATS2015_Training.zip`.
