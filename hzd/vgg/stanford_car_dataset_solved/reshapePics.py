import os
from PIL import Image


def visit_dir(path):
    for root, dirs, files in os.walk(path):
        for filespath in files:
            file_path = os.path.join(root, filespath)
            im = Image.open(file_path)
            im.resize((224, 224), Image.ANTIALIAS).convert('RGB').save(file_path)


if __name__ == "__main__":
    path = "test"
    visit_dir(path)
