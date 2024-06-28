import os
import shutil

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_file(content, folder_path, filename):
    create_folder(folder_path)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path

def save_text_file(text, folder_path, filename):
    create_folder(folder_path)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "w") as f:
        f.write(text)
    return file_path
