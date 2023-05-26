# ClipCap Evolved

## Requirements ##
- Java 1.8.0
- Python 3
- pip packages listed in _requirements.txt_
- wandb account

## Datasets ##
Files to download:
- [Karpathy split annotation file](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip) (only annotation file is required)
- [COCO 2014 (train+val)](https://cocodataset.org/#download)
- [nocaps](https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json)

_generate_coco.py_ should be used to generate proper train/val/test split out of COCO 2014 dataset. It requires annotation file and COCO 2014 dataset to be downloaded and extracted.

_generate_nocaps.py_ should be used to download the validation set of nocaps dataset. It requires nocaps dataset to be downloaded and extracted.

Training is done by executing _train.py_ and specifying proper arguments. List of arguments can be obtained with _python train.py --help_.

Project relies on _wandb_ so the user may be asked to log in the first time they execute the script.
It is also possible to set --offline flag to skip logging to wandb.

List of experiment runs with all arguments can be found in the _start.sh_ file.
