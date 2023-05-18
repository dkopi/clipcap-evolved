import io
import os
import json
import img2dataset
import shutil
import argparse
import requests

def download_json(url, folder):
    filename = url.split('/')[-1]
    filepath = folder + '/' + filename
    if not os.path.exists(filepath):
        response = requests.get(url)
        with open(filepath, 'w') as f:
            f.write(response.text)


def get_captions(annotation_file):
    with io.open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)['annotations']
    captions = []
    for annotation in annotations:
        captions.append(annotation['caption'])
    return captions

def get_images_url(annotation_file):
    with io.open(annotation_file, 'r', encoding='utf-8') as f:
        images = json.load(f)['images']
    images_url = []
    for image in images:
        images_url.append(image['coco_url'])
    return images_url

#We need a txt file in order to use the img2dataset package
def write_urls(url_list):
    with open('data/nocaps/annotations/myimglist.txt', 'w') as f:
        for url in url_list:
            f.write(url + '\n')


def download_dataset(annotation_file: str ,url_list_file: str = "data/nocaps/annotations/myimglist.txt", output_folder: str = "data/nocaps"):
    
    download_json("https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json", "data/nocaps/annotations")
    write_urls(get_images_url(annotation_file))

    img2dataset.download(
        url_list=url_list_file,
        output_folder=output_folder,
        disable_all_reencoding = True,
        timeout=60,
        retries=2,
    )

    urls = get_images_url(annotation_file)

    #renaming the images if they don't have the proper name
    for i, url in enumerate(urls):
      url = url.strip()
      filename = os.path.splitext(url.split('/')[-1])[0]
      src = f'data/nocaps/00000/{i:09d}.jpg'
      jsdel= f'data/nocaps/00000/{i:09d}.json'
      dst = f'data/nocaps/00000/{filename}.jpg'
      print(src, dst)
      if os.path.exists(src):
            os.rename(src, dst)
            os.remove(jsdel)
      else:
            print(f"Error: File {src} does not exist")

    #and finally creating 3 different folders, in-domain, near-domain and out-domain
    path = 'data/nocaps'
    imagepath= 'data/nocaps/00000'
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    images = data['images']
    for image in images:
        domain = image['domain']
        domain_folder = path + '/' + domain
        if not os.path.exists(domain_folder):
            os.makedirs(domain_folder)
        image_path = imagepath  + '/' + image['file_name']
        new_image_path = domain_folder  + '/' + image['file_name']
        if os.path.exists(image_path):
            shutil.move(image_path, new_image_path)
        else:
            print(f"Error: File {image_path} does not exist")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path to annotations file
    parser.add_argument("--annotation_file", default='data/nocaps/annotations/nocaps_val_4500_captions.json')
    args = parser.parse_args()

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    annotation_file = args.annotation_file

    download_dataset(annotation_file)