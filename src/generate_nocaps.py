import io
import os
import json
import img2dataset
import shutil
import argparse
import requests


def download_json(url, folder):
    filename = url.split("/")[-1]
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        response = requests.get(url)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(filepath, "w") as f:
            f.write(response.text)
    else:
        print(f"File {filename} already exists.")
        exit(0)


def split_json_by_domain(data, root_dir):
    in_domain = {
        "licenses": data["licenses"],
        "info": data["info"],
        "images": [],
        "annotations": [],
    }
    near_domain = {
        "licenses": data["licenses"],
        "info": data["info"],
        "images": [],
        "annotations": [],
    }
    out_domain = {
        "licenses": data["licenses"],
        "info": data["info"],
        "images": [],
        "annotations": [],
    }
    all = {
        "licenses": data["licenses"],
        "info": data["info"],
        "images": [],
        "annotations": [],
    }

    # Creating a dictionary for fast lookup
    image_id_to_domain = {}
    for image in data["images"]:
        image_id_to_domain[image["id"]] = image["domain"]
        # if image["domain"] == "in-domain":
        #     in_domain["images"].append(image)
        # elif image["domain"] == "near-domain":
        #     near_domain["images"].append(image)
        # elif image["domain"] == "out-domain":
        #     out_domain["images"].append(image)
        all["images"].append(image)

    for annotation in data["annotations"]:
        domain = image_id_to_domain.get(annotation["image_id"])
        # if domain == "in-domain":
        #     in_domain["annotations"].append(annotation)
        # elif domain == "near-domain":
        #     near_domain["annotations"].append(annotation)
        # elif domain == "out-domain":
        #     out_domain["annotations"].append(annotation)
        all["annotations"].append(annotation)

    def save_json(data, domain):
        filepath = os.path.join(root_dir, "annotations", f"{domain}.json")
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                json.dump(data, f)

    # save_json(in_domain, "in-domain")
    # save_json(near_domain, "near-domain")
    # save_json(out_domain, "out-domain")
    save_json(all, "all")


def get_captions(annotation_file):
    with io.open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    captions = []
    for annotation in annotations:
        captions.append(annotation["caption"])
    return captions


def get_images_url(annotation_file):
    with io.open(annotation_file, "r", encoding="utf-8") as f:
        images = json.load(f)["images"]
    images_url = []
    for image in images:
        images_url.append(image["coco_url"])
    return images_url


# We need a txt file in order to use the img2dataset package
def write_urls(url_list, filepath):
    with open(filepath, "w") as f:
        for url in url_list:
            f.write(url + "\n")


def download_dataset(annotation_file, root_dir):
    url_list_file = os.path.join(root_dir, "annotations", "myimglist.txt")
    output_folder = root_dir

    download_json(
        "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
        os.path.join(root_dir, "annotations"),
    )
    write_urls(get_images_url(annotation_file), url_list_file)

    img2dataset.download(
        url_list=url_list_file,
        output_folder=output_folder,
        disable_all_reencoding=True,
        timeout=60,
        retries=2,
    )

    urls = get_images_url(annotation_file)

    # renaming the images if they don't have the proper name
    for i, url in enumerate(urls):
        url = url.strip()
        filename = os.path.splitext(url.split("/")[-1])[0]
        src = os.path.join(root_dir, "00000", f"{i:09d}.jpg")
        jsdel = os.path.join(root_dir, "00000", f"{i:09d}.json")
        dst = os.path.join(root_dir, "00000", f"{filename}.jpg")
        # print(src, dst)
        if os.path.exists(src):
            os.rename(src, dst)
            os.remove(jsdel)
        else:
            print(f"Error: File {src} does not exist")

    # and finally creating 3 different folders, in-domain, near-domain and out-domain
    with open(annotation_file, "r") as f:
        data = json.load(f)
    images = data["images"]
    for image in images:
        # domain = image["domain"]
        domain = "all"
        domain_folder = os.path.join(root_dir, domain)
        domain_folder = os.path.join(root_dir, domain)
        if not os.path.exists(domain_folder):
            os.makedirs(domain_folder)
        image_path = os.path.join(root_dir, "00000", image["file_name"])
        new_image_path = os.path.join(domain_folder, image["file_name"])
        if os.path.exists(image_path):
            shutil.move(image_path, new_image_path)
        else:
            print(f"Error: File {image_path} does not exist")

    # at the end, creating 3 different json annotations files
    split_json_by_domain(data, root_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to annotations file
    parser.add_argument(
        "--annotation_file", default="annotations/nocaps_val_4500_captions.json"
    )
    parser.add_argument("--root_dir", default="data/nocaps")
    args = parser.parse_args()

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    annotation_file = os.path.join(args.root_dir, args.annotation_file)

    download_dataset(annotation_file, args.root_dir)
