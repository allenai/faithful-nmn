# faithful-nmn
This repository contains the code associated with the paper [Obtaining Faithful Interpretations from Compositional Neural Networks](https://arxiv.org/abs/2005.00724), published at ACL 2020.

## Dependencies
PyTorch 1.2.0 is a dependency. Please see the ```requirements.txt``` file for the other dependencies. Note that the required AllenNLP version is 0.9.0. Also, significant amounts of code were reused from [LXMERT](https://github.com/airsplay/lxmert) and [Learning to Count Objects in Natural Images for Visual Question Answering](https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/counting.py)

## Setup Steps
We recommend the following steps to get started using this repository.
1. Create a new Python environment. Using [anaconda](https://docs.anaconda.com/anaconda/install/), you can do this with ```conda create -n faithful_nmn python=3.6 pip```
2. Install the dependencies. First, follow the [instructions on the PyTorch website](https://pytorch.org/get-started/locally/) to install PyTorch with the correct CUDA version. Then run ```pip install -r requirements.txt```.
3. Download the necessary data and put it in the ```dataset/nlvr2/``` directory (see next section).
4. Download the LXMERT pre-trained model:
```
wget --no-check-certificate https://nlp1.cs.unc.edu/data/model_LXRT.pth -P lxmert_pretrained/
```

## NLVR2 Data preparation
Please download the following files and put them in the ```dataset/nlvr2/``` directory. They were created by the LXMERT authors (or by running the LXMERT data preparation code).
```
wget https://faithful-nmn.s3-us-west-2.amazonaws.com/train.json dataset/nlvr2/
wget https://faithful-nmn.s3-us-west-2.amazonaws.com/valid.json dataset/nlvr2/
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/train_obj36.zip dataset/nlvr2/
unzip dataset/nlvr2/train_obj36.zip -d dataset/nlvr2/ && rm dataset/nlvr2/train_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/valid_obj36.zip dataset/nlvr2/
unzip dataset/nlvr2/valid_obj36.zip -d dataset/nlvr2/ && rm dataset/nlvr2/valid_obj36.zip
```

## Pre-trained models
* [LayerCount NMN](https://faithful-nmn.s3-us-west-2.amazonaws.com/layercount_nmn_model.tar.gz)

* [SumCount NMN](https://faithful-nmn.s3-us-west-2.amazonaws.com/sumcount_nmn_model.tar.gz)

* [CountModule NMN](https://faithful-nmn.s3-us-west-2.amazonaws.com/countmodule_nmn_model.tar.gz)

* [CountModule+Decontextualization NMN](https://faithful-nmn.s3-us-west-2.amazonaws.com/countmodule_decontextualization_nmn_model.tar.gz)

* [CountModule+Pretraining NMN](https://faithful-nmn.s3-us-west-2.amazonaws.com/countmodule_pretraining_nmn_model.tar.gz)

* [CountModule+Pretraining+Decontextualization NMN](https://faithful-nmn.s3-us-west-2.amazonaws.com/countmodule_pretraining_decontextualization_nmn_model.tar.gz)

## Running pre-trained model
To get model predictions on the validation set, use the following command
```
allennlp predict MODEL_URL dataset/nlvr2/valid.json --include-package lib --include-package predictors --use-dataset-reader --dataset-reader validation --output-file predictions.json --predictor full --cuda-device 0
```
, where MODEL_URL should be replaced with the URL for the model that you want to use to obtain predictions. For instance, for the CountModule+Pretraining+Decontextualization NMN, the command should be
```
allennlp predict https://faithful-nmn.s3-us-west-2.amazonaws.com/countmodule_pretraining_decontextualization_nmn_model.tar.gz dataset/nlvr2/valid.json --include-package lib --include-package predictors --use-dataset-reader --dataset-reader validation --output-file predictions.json --predictor full --cuda-device 0
```

## Evaluating pre-trained model
To evaluate model for **accuracy** on the validation set, use the following command.
```
allennlp evaluate model.tar.gz dataset/nlvr2/valid.json --include-package lib
```
To evaluate model for **faithfulness** on the portion of the validation set with annotated module outputs, use the following command.
```
allennlp evaluate model.tar.gz dataset/nlvr2/valid.json --include-package lib -o "{"validation_dataset_reader": {"box_annotations_path": "\"dataset/nlvr2/nlvr2_test_box_annotations.csv\"", "only_with_box_annotation": "true"}}"
```

## Annotation preparation
We provide annotations in the format necessary for our models in ```dataset/nlvr2/all_annotations_round4.tsv```. If you would like to generate these annotations yourself from the QDMR annotations, please run the following commands:
```
cd dataset/nlvr2
python process_raw_data_scripts/process_annotations.py nlvr2_qdmr_programs_11k.csv
python process_raw_data_scripts/filter_annotations.py train_annotations_from_qdmr_rawnumber.tsv dev_annotations_from_qdmr_rawnumber.tsv test1_annotations_from_qdmr_rawnumber.tsv simple_count_annotations.tsv
mv filtered_annotations.tsv all_annotations.tsv
```
Note that the resulting file will not be identical to ```dataset/nlvr2/all_annotations_round4.tsv``` because it will include test annotations and around 200 extra programs for simple count questions that are in ```simple_count_annotations.tsv```.

## Training a model
To train a model, please use the following command:
```allennlp train train_configs/nlvr2.jsonnet -s serialized_dir --include-package lib```


# DROP Experiments

Data: `dataset/drop` contains the manual annotations for intermediate module outputs for DROP.

Code: [Text-NMN-faithful](https://github.com/nitishgupta/nmn-drop/tree/interpret) -- Model code, evaluation scripts, etc. can be found here.

# Citation
If you find our work relevant/useful to yours, please cite our paper:
```
@inproceedings{subramanian-etal-2020-obtaining,
    title = "Obtaining Faithful Interpretations from Compositional Neural Networks",
    author = "Subramanian, Sanjay  and
      Bogin, Ben  and
      Gupta, Nitish  and
      Wolfson, Tomer  and
      Singh, Sameer  and
      Berant, Jonathan  and
      Gardner, Matt",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.495",
    pages = "5594--5608"
}
```
