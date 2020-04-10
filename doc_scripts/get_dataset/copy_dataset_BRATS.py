import os
import argparse
import zipfile
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--struct", help="save original struct", action="store_true")
parser.add_argument("-p", "--path", default='/root/BRATS/', help="path to save dataset")
parser.add_argument("-t", "--tar_name", help="name of untar file")
parser.add_argument("--type", choices=['HGG', 'LGG', 'ALL'], default='ALL', help="type of brain tumor")
args = parser.parse_args()

with zipfile.ZipFile(os.path.join(args.path, args.tar_name), 'r') as zipObj:
    tmp_dir = '/root/tmp'
    try:
        os.mkdir(tmp_dir)
        if not args.struct:
            if args.type in ['ALL', 'HGG']:
                os.mkdir(os.path.join(args.path, 'HGG'))
            if args.type in ['ALL', 'LGG']:
                os.mkdir(os.path.join(args.path, 'LGG'))
    except Exception as e:
        print(e)
    listOfFileNames = zipObj.namelist()
    for fileName in listOfFileNames:
        name_list = fileName.split('/')
        if ((args.type == 'ALL') or name_list[1] == args.type) and fileName.endswith('.mha'):
            img_type = name_list[-1].split('.')[-3]
            if img_type == 'MR_T2' or img_type == 'OT':
                if args.struct:
                    zipObj.extract(fileName, args.path)
                else:
                    zipObj.extract(fileName, tmp_dir)
                    shutil.move(os.path.join(tmp_dir, fileName), os.path.join(args.path, f'{name_list[1]}', f'{name_list[2].split("_")[2]}_{"tumor" if img_type == "OT" else "brain"}.mha'))
    if not args.struct:
        shutil.rmtree(tmp_dir)
