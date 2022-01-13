# Check for and extract data files

from pathlib import Path
from zipfile import ZipFile

def unzipper(file_names, zip_folder, path):
    for file in file_names:
        if Path(path + file).exists():
            print(file + ' already exists')
        else:
            if '.zip' in zip_folder:
                with ZipFile(path + '/' + zip_folder, 'r') as zip:
                    zip.extractall(path)
            else:    
                with ZipFile(path + '/' + zip_folder + '.zip', 'r') as zip:
                    zip.extractall(path)
            print(file + ' extracted')