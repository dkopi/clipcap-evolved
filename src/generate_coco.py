import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/dataset.json")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    dataset = json.load(open(args.dataset_path))

    val_images = []
    val_captions = []
    train_images = []
    train_captions = []
    test_images = []
    test_captions = []
    for anno in dataset["images"]:
        image = {
            "id": anno["imgid"],
            "file_name": os.path.join(anno["filepath"], anno["filename"]),
        }
        if anno["split"] == "val":
            val_images.append(image)
        elif anno["split"] == "train" or anno["split"] == "restval":
            train_images.append(image)
        elif anno["split"] == "test":
            test_images.append(image)

        for sentence in anno["sentences"]:
            caption = {
                "image_id": anno["imgid"],
                "id": sentence["sentid"],
                "caption": sentence["raw"],
            }
            if anno["split"] == "val":
                val_captions.append(caption)
            elif anno["split"] == "train" or anno["split"] == "restval":
                train_captions.append(caption)
            elif anno["split"] == "test":
                test_captions.append(caption)

    print(f"val_images: {len(val_images)}")
    print(f"val_captions: {len(val_captions)}")
    print(f"train_images: {len(train_images)}")
    print(f"train_captions: {len(train_captions)}")
    print(f"test_images: {len(test_images)}")
    print(f"test_captions: {len(test_captions)}")

    train = {
        "images": train_images,
        "annotations": train_captions,
    }
    val = {
        "images": val_images,
        "annotations": val_captions,
    }
    test = {
        "images": test_images,
        "annotations": test_captions,
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(train, f)

    with open(os.path.join(args.output_dir, "val.json"), "w") as f:
        json.dump(val, f)

    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(test, f)
