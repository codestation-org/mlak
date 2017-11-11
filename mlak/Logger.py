#! /usr/bin/python3
"""
Created on Sat Nov 11 2017
author: Wojciech Peisert

Logger for data and files (it's hashes)
"""

import subprocess
import hashlib
import shlex
import time
import datetime
import os
import json
from collections import OrderedDict

class Logger:

	default_log_file_name = "default.log"
	default_log_file_ext  = "log"
	default_log_dir       = "logs/"
	default_split         = False

	#log(data, files, log_file_name="default.log", log_file_ext="log", log_dir="logs/", split=False):
	def log(data, files, **kwArgs):

		log_file_name = kwArgs.get("log_file_name", Logger.default_log_file_name)
		log_file_ext  = kwArgs.get("log_dir", Logger.default_log_file_ext)
		log_dir       = kwArgs.get("log_dir", Logger.default_log_dir)
		split         = kwArgs.get("split", Logger.default_split)

		os.makedirs(log_dir, exist_ok = True)

		if split:
			log_file_name = Logger.get_unique_log_filename() + '.' + log_file_ext
			
		log_full_path = log_dir + log_file_name
		
		json = Logger.get_json(data, files) + '\n'

		if split:
			write_to_file(log_full_path, json)
		else:
			append_to_file(log_full_path, json)


	def get_json(data, files):
		gitcurrenthash, gitchanges = get_git_hash_and_changes()	
		jsonContent = OrderedDict([
			('date', get_current_datetime()),
			('git', OrderedDict([
					('currenthash', gitcurrenthash),
					('changes',     gitchanges)
				])
			),
			('data', data),
			('files', Logger.get_files_list(files))
		])
		return json.dumps(jsonContent).replace('\n', ' \\n ')

	def get_files_list(files):
		fileList = []
		for filename in files:
			fileList.append(
				OrderedDict([
					('filename', filename),
					('hash',     file_checksum_md5(filename))
				])
			)
		return fileList

	def get_unique_log_filename():
		txt1 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
		txt2 = hashlib.md5(str(time.time()).encode()).hexdigest()
		return txt1 + '-' + txt2[:6]


def write_to_file(path, content):
	obj = open(path, 'wb')
	obj.write(content.encode())
	obj.close


def append_to_file(path, content):
	obj = open(path, 'ab')
	obj.write(content.encode())
	obj.close


def get_current_datetime():
	return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')


def file_checksum_md5(filename):
    md5 = hashlib.md5()
    with open(filename,'rb') as f: 
        for chunk in iter(lambda: f.read(8192), b''): 
            md5.update(chunk)
    return md5.hexdigest()


def get_git_hash_and_changes():
	currenthash = subprocess.check_output(shlex.split('git rev-parse HEAD'))
	changes = subprocess.check_output(shlex.split('git diff --shortstat'))
	return (currenthash.decode().replace('\n', ''), changes.decode().replace('\n', ''))
