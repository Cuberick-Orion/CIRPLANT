## Installation

### Requirements
- Python >= 3.7
- Pytorch >= 1.6
- torchvision >= 0.4.0
- Pytorch Lightning >= 1.3.1
- cuda >= 10.0

>Lower versions are potentially okay, but untested.

### Step 1: Install OSCAR

Follow the following instructions from [OSCAR INSTALL.md](https://github.com/microsoft/Oscar/blob/master/INSTALL.md) with some modifications.

```bash
# create a new environment
conda create --name cirr python=3.7
conda activate cirr

# install pytorch (latest version)
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install oscar
cd $INSTALL_DIR
git clone --recursive git@github.com:microsoft/Oscar.git
cd Oscar/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt

unset INSTALL_DIR
```

>`apex` is optional, as Pytorch Lightning uses native Automatic Mixed Precision.

>If you encounter warnings on CUDA version when installing `apex`, follow the prompt to comment it out and proceed.

### Step 2: Setup CIRPLANT
```bash
# clone this repository
git clone git@github.com:Cuberick-Orion/CIRPLANT.git

# install additional required packages
conda activate cirr
cd CIRPLANT
pip install -r requirements.txt
cd ..
```

### Step 3.1: Download pre-trained OSCAR models
The pre-trained OSCAR models can be downloaded [here](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md#pre-trained-models). We use the `base-vg-labels`.

Download the pre-trained weights to `data/Oscar_pretrained_models`.

```bash
# create data directory
cd CIRPLANT
mkdir data
cd data

# create folder for pre-trained OSCAR model
mkdir Oscar_pretrained_models
cd Oscar_pretrained_models

# download the pre-trained weights
wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-vg-labels.zip
# ! alternatively, use AzCopy (will be much faster)
# path/to/azcopy copy <URL> <local_path>

# unzip zip file into data folder (see below for file structure)
# then delete it
unzip base-vg-labels.zip -d $DATA_DIR
rm base-vg-labels.zip
```
>The unzipped files should look like:
>```
> data
> └─── Oscar_pretrained_models
>      └─── base-vg-labels
>           ├── ep_67_588997      
>           │           ...
>           └── ep_107_1192087         
>                       ...
>```

### Step 3.2: Prepare datasets
 - For CIRR, please see our [dataset repository](https://github.com/Cuberick-Orion/CIRR). Download the annotations and image features to `data/cirr`.
 - For other datasets, we recommend following a similar file structure and save to `data/$DATASET_NAME`.

>See the CIRR dataset [file structure](https://github.com/Cuberick-Orion/CIRR/blob/main/README.md#dataset-file-structure) for what it should look like.