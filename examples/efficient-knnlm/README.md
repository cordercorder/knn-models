# Efficient Nearest Neighbor Language Models

[https://aclanthology.org/2021.emnlp-main.461.pdf](https://aclanthology.org/2021.emnlp-main.461.pdf)


The paper [Efficient Nearest Neighbor Language Models](https://aclanthology.org/2021.emnlp-main.461.pdf) 
proposes to apply Greedy Merging and PCA to the datastore to reduce the inference overhead introduced by the 
retrieval operation. We provide instructions on how to use them with kNN-models on this page, taking 
Wikitext-103 as an example.


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
in the paper, we set the captured module to the layer norm 
before FFN of the last layer (`--module-to-capture "layers[-1].final_layer_norm"`). 
**Note that this step will consume about 200GB of disk space. Please 
ensure there is enough free disk space before running this.**

``` bash
knn_models=/path/to/knn_models
wikitext_data_bin=/path/to/wikitext-103-dataset-databin
checkpoint=/path/to/pretrained-model/model.pt
datastore=/path/to/datastore

mkdir -p ${datastore}

# there are 103227021 tokens in the training set of wikitext-103, 
# since the first sample need to be skipped due to its incomplete 
# context window, the datastore size is 1536 tokens less than 103227021
datastore_size=103225485

generate_lm_datastore ${wikitext_data_bin} \
    --user-dir ${knn_models} \
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

## Dimension reduction
The script below applies PCA to the feature dimension of the datastore keys. 
The datastore keys after processing using PCA and the [PCA matrix](https://github.com/facebookresearch/faiss/wiki/Python-C---code-snippets#how-can-i-get-the-pca-matrix-in-numpy-from-a-pca-object) 
will save to the directory specified by `transformed_datastore`. 
`reduced_keys_dimension` denotes the number of the feature dimensions to keep, 
which is set to `512` in this case.
``` bash
datastore=/path/to/datastore
datastore_size=103225485
reduced_keys_dimension=512
transformed_datastore=/path/to/transformed-datastore

mkdir -p ${transformed_datastore}

reduce_datastore_dims \
    --method PCA \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --transformed-datastore ${transformed_datastore} \
    --reduced-keys-dimension ${reduced_keys_dimension} \
    --random-rotation

# as the datastore values remain unchanged in this step,
# create a soft link between them to avoid copy
ln -s ${datastore}/values.npy ${transformed_datastore}/values.npy
```

## Build the Faiss index

Due to the memory constraint (a server with 256 GB RAM), we only randomly 
sample 40000000 keys to train the Faiss index. The number of sampled keys
can be adjusted according to the hardware environment. This step will 
construct `IndexIVFPQ` with Faiss and save it as a file named `faiss.index` 
to the `transformed_datastore` directory. 

``` bash
transformed_datastore=/path/to/transformed-datastore
reduced_keys_dimension=512
datastore_size=103225485

num_trained_keys=40000000

build_faiss_index \
    --use-gpu \
    --datastore ${transformed_datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension ${reduced_keys_dimension} \
    --num-trained-keys ${num_trained_keys} \
    --keys-dtype fp16 \
    --knn-fp16
```


## Prune datastore
Apply greedy merging to prune redundant records in the datastore:

``` bash
transformed_datastore=/path/to/transformed-datastore
pruned_datastore=/path/to/pruned-datastore
datastore_size=103225485
reduced_keys_dimension=512
num_neighbors=30

mkdir -p ${pruned_datastore}

prune_datastore \
    --method greedy_merge \
    --datastore ${transformed_datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension ${reduced_keys_dimension} \
    --pruned-datastore ${pruned_datastore} \
    --use-gpu \
    --num-neighbors ${num_neighbors} \
    --knn-fp16
```


## Evaluation
There is a weight for each value of the datastore after 
applying greedy merging, which is used to remedy the information 
loss during pruning. Consequently, different from the common evaluation 
setting of [kNN-LM](../knnlm/README.md), the script below loads 
the value weights (`--load-value-weights`).

``` bash
knn_models=/path/to/knn_models
wikitext_data_bin=/path/to/wikitext-103-dataset-databin
pruned_datastore=/path/to/pruned-datastore
checkpoint=/path/to/pretrained-model/model.pt
reduced_keys_dimension=512
datastore_size="pruned datastore size"

eval_knn_lm ${wikitext_data_bin} \
    --user-dir ${knn_models} \
    --task language_modeling_knn \
    --module-to-capture "layers[-1].final_layer_norm" \
    --recompute-distance \
    --load-keys \
    --load-value-weights \
    --keys-dimension ${reduced_keys_dimension} \
    --datastore ${pruned_datastore} \
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

