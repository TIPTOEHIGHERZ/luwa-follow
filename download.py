#! /bin/python3
import os
import argparse


pswd = 'ustc.112233'
ip = 'user@192.168.51.24:'
base_dir = '/mnt/share/debug/zzy/'

parser = argparse.ArgumentParser('upload')
# parser.add_argument('-f', '--file', help='files to upload to server', type=str)
parser.add_argument('file', help='files to upload to server', type=str)
parser.add_argument('-d', '--dst', help='destination for the file', type=str)
parser.add_argument('-r', '--recursive', help='upload files recursively', action='store_true')
args = parser.parse_args()

file_name = ip + base_dir + args.file

dst = args.dst if args.dst else './checkpoints/'
if dst[-1] != '/':
    dst += '/'

if args.recursive:
    os.system(f'sshpass -p \'{pswd}\'  rsync -av --progress {file_name} {dst}')
else:
    os.system(f'sshpass -p \'{pswd}\' rsync -v --progress {file_name} {dst}')
