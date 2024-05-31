import json
import glob
import re
from datetime import datetime

def read_json(filepath):
    with open(filepath, 'r', encoding='UTF-8') as file:
        return json.load(file)
    
def save_json(filepath, data):
    with open(filepath, 'w', encoding='UTF-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def find_all_files(dir, pattern='', ending='.json'):
    # Use glob.glob() to find all file paths in the directory ending with ".json"
    filepaths = glob.glob(dir + '/**/*' + ending, recursive=True)
    # Filter the file paths using re.match() and the exclude pattern
    return [filepath for filepath in filepaths if re.match(pattern, filepath)]

def read_text(filepath):
    with open(filepath, 'r', encoding='UTF-8') as f:
        article = f.read()
    return article

def save_text(filepath, data):
    with open(filepath, 'w', encoding='UTF-8') as file:
        file.write(str(data))

def read_lines(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return lines

def current_date():
    now = datetime.now()
    return str(now.strftime('%Y-%m-%d %H:%M:%S'))

def count_words(text):
    words = text.split()
    return len(words)
