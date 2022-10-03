# Generalization through Memorization: Nearest Neighbor Language Models

[https://openreview.net/pdf?id=HklBjCEKvH](https://openreview.net/pdf?id=HklBjCEKvH)


This page provides instructions on how to reproduce kNN-LM with kNN-models, taking 
Wikitext-103 as example.


## Preprocess the data

Download the Wikitext-103 dataset:
``` bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```

Binarize the data with `fairseq-preprocess`:
``` bash
wikitext=/path/to/wikitext-103-dataset
wikitext_data_bin=/path/to/wikitext-103-dataset-databin

fairseq-preprocess \
    --only-source \
    --trainpref ${wikitext}/wiki.train.tokens \
    --validpref ${wikitext}/wiki.valid.tokens \
    --testpref ${wikitext}/wiki.test.tokens \
    --destdir ${wikitext_data_bin} \
    --workers 20
```


## Download the pre-trained model

Download the pre-trained model of [(Baevski and Auli, 2018)	](https://arxiv.org/abs/1809.10853):
``` bash
wget https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2
tar -xjvf adaptive_lm_wiki103.v2.tar.bz2
```


## Generate the datastore
Generate the datastore with the pre-trained language model. 

Since the optimal intermediate hidden states that serve as the 
datastore keys for wikitext-103 dataset on language modeling task 
are the FFN input after layer norm according to ablation experiments 
in the original paper, we set the captured module to the layer norm 
before FFN of the last layer (`--module-to-capture "layers[-1].final_layer_norm"`). 
**Note that this step will consume about 200GB of disk space. Please 
ensure there is enough free disk space before running this.**

``` bash
wikitext_data_bin=/path/to/wikitext-103-dataset-databin
checkpoint=/path/to/pretrained-model/model.pt
datastore=/path/to/datastore

mkdir -p ${datastore}

# there are 103227021 tokens in the training set of wikitext-103, 
# since the first sample need to be skipped due to its incomplete 
# context window, the datastore size is 1536 tokens less than 103227021
datastore_size=103225485

generate_lm_datastore ${wikitext_data_bin} \
    --task language_modeling_knn \
    --module-to-capture "layers[-1].final_layer_norm" \
    --sample-break-mode none \
    --max-tokens 3072 \
    --softmax-batch 1024 \
    --context-window 1536 \
    --tokens-per-sample 3072 \
    --gen-subset train \
    --path ${checkpoint} \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dtype fp16 \
    --saving-mode
```

## Build the Faiss index

Due to the memory constraint (a server with 256 GB RAM), we only randomly 
sample 40000000 keys to train the Faiss index. The number of sampled keys
can be adjusted according to the hardware environment. This step will 
construct `IndexIVFPQ` with Faiss and save it as a file named `faiss.index` 
to the `datastore` directory.

``` bash
datastore=/path/to/datastore

datastore_size=103225485

num_trained_keys=40000000

build_faiss_index \
    --use-gpu \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --num-trained-keys ${num_trained_keys} \
    --keys-dtype fp16 \
    --knn-fp16
```

## Evaluation

As the `IndexIVFPQ` returns approximate distances, the original implementation 
of kNN-LM recomputes the distance with the retrieved keys to obtain better 
performance. We follow it in the shell script below to reproduce the 
experiment results reported in the paper (`--recompute-distance`). However, 
recomputing the distance needs to frequently load the keys from the 
disk (`--load-keys`) during inference, which will slow down the inference 
speed due to the heavy data transfer between the CPU memory and disk. To 
speed up the inference, we move the keys from the disk to the CPU memory 
in advance (`--move-to-memory`), which will incur memory overhead instead. 
Additionally, we place the Faiss index on another GPU device to avoid 
out of memory (`--knn-device-id 1`). The `--recompute-distance`, 
`--load-keys`, and `--move-to-memory` arguments can be removed if there is 
not enough RAM on your server, which will trade off a slightly worse result.

``` bash
wikitext_data_bin=/path/to/wikitext-103-dataset-databin
datastore=/path/to/datastore
checkpoint=/path/to/pretrained-model/model.pt
datastore_size=103225485

eval_knn_lm ${wikitext_data_bin} \
    --task language_modeling_knn \
    --module-to-capture "layers[-1].final_layer_norm" \
    --recompute-distance \
    --load-keys \
    --keys-dimension 1024 \
    --datastore ${datastore} \
    --move-to-memory \
    --datastore-size ${datastore_size} \
    --knn-device-id 1 \
    --num-neighbors 1024 \
    --lambda-value 0.25 \
    --score-knn-lm \
    --knn-fp16 \
    --num-workers 0 \
    --path ${checkpoint} \
    --sample-break-mode complete \
    --max-tokens 3072 \
    --tokens-per-sample 3072 \
    --context-window 2560 \
    --softmax-batch 1024 \
    --gen-subset test
```
