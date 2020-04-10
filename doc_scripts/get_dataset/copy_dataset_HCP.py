import os
from tqdm import tqdm
import pandas
import boto3
import botocore
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--struct", help="save original struct",
                    action="store_true")
parser.add_argument("-p", "--path", default='/root/HCP/',
                    help="path to save dataset")
args = parser.parse_args()

# Init variables
s3_bucket_name = 'hcp-openaccess'
s3_prefix = 'HCP_1200/'
try:
	boto3.setup_default_session(profile_name='hcp')
	s3 = boto3.resource('s3')
	bucket = s3.Bucket('hcp-openaccess')

	out_dir = args.path
	client = boto3.client('s3')
	result = client.list_objects(Bucket=s3_bucket_name, Prefix=s3_prefix, Delimiter='/')
except Exception as e:
	print(e)
else:
	people_list = [int(o.get('Prefix').split('/')[-2]) for o in result.get('CommonPrefixes')]
	people_list = people_list[:10]
	#
	for person in tqdm(people_list):
		s3_keys = bucket.objects.filter(Prefix=f'HCP_1200/{person}/MNINonLinear/T2w_restore')
		s3_keylist = [key.key for key in s3_keys]
		for path_idx, s3_path in enumerate(s3_keylist):
			rel_path = s3_path.replace(s3_prefix, '')
			rel_path = rel_path.lstrip('/')
			if args.struct:
				download_file = os.path.join(out_dir, rel_path)
				download_dir = os.path.dirname(download_file)
			else:
				download_file = os.path.join(out_dir, rel_path.split('/')[-1][:-7],
											 '.'.join([rel_path.split('/')[-1].split('.')[0]
								+ '_'+ str(person)] + rel_path.split('/')[-1].split('.')[1:]))
				download_dir = os.path.dirname(download_file)
			os.makedirs(download_dir, exist_ok=True)
			try:
				if not os.path.exists(download_file):
					with open(download_file, 'wb') as f:
						bucket.download_file(s3_path, download_file)
			except Exception as exc:
				print(f'There was a problem downloading {s3_path}.\n Check and try again.')
				print(exc)
