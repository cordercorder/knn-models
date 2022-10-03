# Nearest Neighbor Machine Translation

[https://openreview.net/pdf?id=7wCBOfJ8hJM](https://openreview.net/pdf?id=7wCBOfJ8hJM)


This page provides instructions on how to reproduce kNN-MT with kNN-models. 
We use the [multi-domain parallel corpus](https://github.com/roeeaharoni/unsupervised-domain-clusters) 
of German-English as an example and mainly focus on the IT domain on this page.


## Download the pre-trained model
Download the pre-trained German-English NMT model of 
[(Ng et al., 2019)](https://aclanthology.org/W19-5333.pdf):
``` bash
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.ffn8192.tar.gz
tar -zxvf wmt19.de-en.ffn8192.tar.gz
```

## Preprocess the data

Once the [multi-domain parallel corpus](https://github.com/roeeaharoni/unsupervised-domain-clusters) 
has been downloaded. It can be preprocessed by the `prepare-domadapt.sh` script. Please ensure 
[mosesdecoder](https://github.com/moses-smt/mosesdecoder) has been downloaded and 
[fastBPE](https://github.com/glample/fastBPE) has been compiled before running 
this script as it depends on them to be executed. There are two command line arguments 
in this script, which are the paths of the directories that contain the raw 
and preprocessed parallel corpus respectively. Additionally, the scripting variables 
of `mosesdecoder`, `fastbpe`, and `pretrained_model` should be properly set in your case.


``` bash
domain="it"
multi_domin_corpus=/path/to/multi-domin-corpus
preprocessed_multi_domin_corpus=/path/to/preprocessed-multi-domin-corpus
bash prepare-domadapt.sh ${multi_domin_corpus}/${domain} ${preprocessed_multi_domin_corpus}/${domain}
```


Binarize the data with `fairseq-preprocess`:
``` bash
domain="it"
preprocessed_multi_domin_corpus=/path/to/preprocessed-multi-domin-corpus/${domain}
srcdict=/path/to/pretrained_model/dict.en.txt
data_bin=/path/to/multi-domin-data-bin/${domain}

fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --trainpref ${preprocessed_multi_domin_corpus}/train.bpe.filtered \
    --validpref ${preprocessed_multi_domin_corpus}/dev.bpe \
    --testpref ${preprocessed_multi_domin_corpus}/test.bpe \
    --srcdict ${srcdict} \
    --joined-dictionary \
    --workers 16 \
    --destdir ${data_bin}
```

## Generate the datastore
Generate the datastore with the pre-trained NMT model:
``` bash
domain="it"
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt

mkdir -p ${datastore}

generate_mt_datastore ${data_bin} \
    --task translation_knn \
    --gen-subset train \
    --path ${checkpoint} \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dtype fp16 \
    --max-tokens 8000 \
    --saving-mode
```

## Build the Faiss index
The script below will construct `IndexIVFPQ` with Faiss and save it 
as a file named `faiss.index` to the `datastore` directory.
``` bash
domain="it"
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`

build_faiss_index \
    --use-gpu \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --keys-dtype fp16 \
    --knn-fp16
```

## Tune hyperparameters

The hyperparameters are chosen according to the BLEU score on the validation set. 
The script below will perform grid search on `num_neighbors` ∈ [2, 4, 8, 16, 32, 64], 
`lambda` ∈ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], and `temperature` ∈ 
[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]. 
Note that `tune_knn_params` will detect all the available GPU devices and create 
a subprocess on each GPU device. Please set the environment variable 
`CUDA_VISIBLE_DEVICES` to avoid too many subprocesses.

``` bash
export CUDA_VISIBLE_DEVICES=0

domain="it"
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt

max_tokens=8000

nohup tune_knn_params \
    --reference ${multi_domin_corpus}/${domain}/dev.en \
    --candidate-num-neighbors 2 4 8 16 32 64 \
    --candidate-lambda-value 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --candidate-temperature-value 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100  \
    --sacrebleu-args "-w 6" \
    $(which fairseq-generate) ${data_bin} \
        --task translation_knn \
        --datastore ${datastore} \
        --datastore-size ${datastore_size} \
        --knn-fp16 \
        --source-lang de \
        --target-lang en \
        --gen-subset valid \
        --path ${checkpoint} \
        --max-tokens ${max_tokens} \
        --beam 5 \
        --tokenizer moses \
        --post-process subword_nmt > tune_knn_params.logs 2>&1 &
```

## Evaluation
Translate the test set with kNN-MT:
``` bash
domain="it"

num_neighbors="selected num_neighbors"
lambda="selecteed lambda"
temperature="selected temperature"

knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt

max_tokens=8000

fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_knn \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --lambda-value ${lambda} \
    --temperature-value ${temperature} \
    --source-lang de \
    --target-lang en \
    --gen-subset test \
    --path ${checkpoint} \
    --max-tokens ${max_tokens} \
    --beam 5 \
    --tokenizer moses \
    --post-process subword_nmt > raw_sys.de-en.en

cat raw_sys.de-en.en | grep -P "^D" | sort -V | cut -f 3- > sys.de-en.en
sacrebleu -w 6 ${multi_domin_corpus}/${domain}/test.en --input sys.de-en.en
```
