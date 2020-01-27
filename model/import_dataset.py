import random
import numpy as np
import nibabel as nib
import os
import glob
from funcs import preproc

def get_id(name):
    return int(name.split("_")[-1].split(".")[0])

# hint: вынес 'wd' в параметры
def create_datasets(wd, retrain=False, task=None, labels=False, ds_scale=0):
    # wd = "./Data/CamCAN_unbiased/CamCAN/T2w"
    # os.chdir(wd)

    # subject_id = np.array([i for i in glob.glob(wd+"/*") if '.txt' not in i and 'normalized' in i])
    test_flg = 1
    if test_flg:
        subject_id = np.array([subject for subject in os.listdir(wd)[:5]])
        subject_train_idx = random.sample(range(len(subject_id)), 3)
        subject_train = subject_id[subject_train_idx]
        subject_test = [i for i in subject_id if i not in subject_train]
    else:
        subject_id = np.array([subject for subject in os.listdir(wd)])
        subject_train_idx = random.sample(range(len(subject_id)), 400)
        subject_train = subject_id[subject_train_idx]
        subject_test = [i for i in subject_id if i not in subject_train][:50]


    if not retrain:
        np.savetxt(str(task)+"_subject_train.txt", subject_train, "%s", delimiter=',')
        np.savetxt(str(task)+"_subject_test.txt", subject_test, "%s", delimiter=',')

    else:
        subject_train = np.genfromtxt(str(task)+"_subject_train.txt", dtype=str, delimiter=',')
        subject_test = np.genfromtxt(str(task)+"_subject_test.txt", dtype=str, delimiter=',')

    print(f"retrain is {retrain}")
    # print(str(task)+f"training subject ids: {subject_train}")
    # print(str(task)+f"testing subject ids: {subject_test}")

    X_train_input = []
    X_train_target = []
    X_train_target_all = []
    X_dev_input = []
    X_dev_target = []
    X_dev_target_all = []

    # assert False, 'lol create_dataset'

    for i in subject_train:
        print(i)
        pathx = os.path.join(wd, i)
        img = nib.load(pathx).get_data()
        img = np.transpose(img, [2, 0, 1])
        idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        img = img[idx]
        _, x, y = img.shape
        max_xy = max(x, y)


        # a, b -- не целые, я понятия не имею, как это работало
        # a = (max_xy-x)/2
        # b = (max_xy-y)/2
        a = int((max_xy-x)/2)
        b = int((max_xy-y)/2)

        if len(X_train_input) == 0:
            print(img.shape)

        img = np.pad(img, ((0, 0), ((a, a)), (b, b)), mode='edge')
        img = preproc.resize(img, 256 / 233.0)

        if labels:
            z = np.genfromtxt(os.path.join(os.path.dirname(os.path.dirname(wd)), wd.split('/')[-2]+'_labels', i.split('.')[0] + "_label.txt"))
            assert len(z) == len(idx)
            X_train_target.extend(z)

        if ds_scale != 0:
            img = preproc.downsample_image(img[np.newaxis, :, :, :], ds_scale)
        X_train_input.extend(img)


    for j in subject_test:
        print(j)
        pathx = os.path.join(wd, i)
        img = nib.load(pathx).get_data()
        img = np.transpose(img, [2, 0, 1])
        idx = [s for s in range(img.shape[0]) if len(set(img[s].flatten())) > 1]
        img = img[idx]
        img = np.pad(img, ((0, 0), (a, a), (b, b)), mode='edge')
        img = preproc.resize(img, 256 / 233.0)

        if ds_scale!=0:
            img = preproc.downsample_image(img[np.newaxis,:,:,:],ds_scale)
        X_dev_input.extend(img)
        if labels:
            z = np.genfromtxt(os.path.join(os.path.dirname(os.path.dirname(wd)), wd.split('/')[-2]+'_labels', i.split('.')[0] + "_label.txt"))
            print(len(z), img.shape)
            X_dev_target.extend(z)

    if not labels:
        return np.asarray(X_train_input), np.asarray(X_dev_input)
    else:
        return X_train_input, X_train_target, X_dev_input, X_dev_target