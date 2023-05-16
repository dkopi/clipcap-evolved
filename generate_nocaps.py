import io
import os
import json
import img2dataset

# Path to annotations file
annotation_file = 'data/nocaps/annotations/nocaps_val_4500_captions.json'

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

def download_dataset(url_list_file: str = "data/nocaps/annotations/myimglist.txt", output_folder: str = "data/nocaps"):
    
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
      if os.path.exists(src):
            os.rename(src, dst)
            os.remove(jsdel)
      else:
            print(f"Error: File {src} does not exist")

if __name__ == '__main__':
    download_dataset()