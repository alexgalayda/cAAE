Dear Santa Claus, I have been good this year. For example, I fixed:

* README.md:  
  * python>=2.7, tensorlayer, tensorflow>=1.0 --> python>=3.6, tensorlayer, tensorflow>=2.0  
    1. Before me, in the code were strings ''.format(). This is immediately python>=3.5. So replace with f-string (python>=3.6)  
    2. tensorlayer requires tensorflow>=2.0. So we suffer, but continue to switch to 2.X
* main.py:
  * os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU_NAME' -- no longer needed, set in docker
  * import tensorflow as tf --> import tensorflow.compat.v1 as tf -- We use this so that code written in tf 1.X works in tf 2.0
  * from funcs.preproc import * -- Don't do that
  * results_path = './Results/Adversarial_Autoencoder' --> parser.add_argument('--dataset_dir'), parser.add_argument('--results_path')
  * tf.disable_eager_execution() -- oh, it's fun. `RuntimeError: tf.placeholder() is not compatible with eager execution.` To fix this error, you need to add [this](https://stackoverflow.com/questions/56561734/runtimeerror-tf-placeholder-is-not-compatible-with-eager-execution). 
  * encoder(x_input, reuse=False, is_train=True) --> encoder(x_input, z_dim=z_dim, reuse=False, is_train=True) -- In the syntax of the function in model.py. I do not understand how it worked before.
  
* model.py:
  * import tensorflow.compat.v1 as tf -- generally skipped. I do not understand how it worked before.
  * from tensorlayer.layers import * -- Don't do that
* import_dataset.py:
  * wd = "./Data/CamCAN_unbiased/CamCAN/T2w" -- 'wd' placed in parameters
  
For example, I want an explanation:
* import_dataset.py:  
  a = (max_xy-x)/2, b = (max_xy-y)/2. But np.pad(img, ((0, 0), ((a, a)), (b, b)), mode='edge') requires int type a and b, and they can (and become) float.
* main.py:
  `train() -> datasets.create_datasets(retrain=0, task="aae_wgan_" + str(z_dim), num_aug=0)` and `datasets -> create_datasets(retrain=False, task=None, labels=False, ds_scale=0)`. In which version is the error?
