import os
import shutil
import requests
import zipfile
from tqdm import tqdm
from urllib.parse import urlparse


def download_file(url, output_folder):
    if not os.path.exists(output_folder):
        print("Creating output folder", output_folder)
        os.makedirs(output_folder)

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    basename = os.path.basename(urlparse(url).path)
    filename = os.path.join(output_folder, basename)

    print("Downloading", basename)
    with open(filename, 'wb') as file, tqdm(
            desc=basename,
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return filename


def extract_zip(zip_file, destination):
    try:
        os.makedirs(destination)
        print("Creating destination folder", destination)
    except OSError:
        pass

    basename = os.path.basename(zip_file)
    print("Extracting", basename)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination)


def delete_folder_with_contents(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")


wider_face_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip?download=true"
wider_face_zip = download_file(wider_face_url, "temp")
extract_zip(wider_face_zip, "resources/wider_face")

wider_face_train_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip?download=true"
wider_face_train_zip = download_file(wider_face_train_url, "temp")
extract_zip(wider_face_train_zip, "resources/wider_face")

wider_face_val_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip?download=true"
wider_face_val_zip = download_file(wider_face_val_url, "temp")
extract_zip(wider_face_val_zip, "resources/wider_face")

wider_face_test_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip?download=true"
wider_face_test_zip = download_file(wider_face_test_url, "temp")
extract_zip(wider_face_test_zip, "resources/wider_face")

delete_folder_with_contents("temp")
