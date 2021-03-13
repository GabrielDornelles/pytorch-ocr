import os

rootdir = 'dataset/'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if len(file) != 9:
            os.remove(f"{rootdir}/{file}")
            print(f"removed {os.path.join(subdir, file)}")