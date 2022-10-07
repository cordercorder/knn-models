# Efficient Cluster-Based k-Nearest-Neighbor Machine Translation

[https://aclanthology.org/2022.acl-long.154.pdf](https://aclanthology.org/2022.acl-long.154.pdf)


The paper [Efficient Cluster-Based k-Nearest-Neighbor Machine Translation](https://aclanthology.org/2022.acl-long.154.pdf) 
proposes two approaches: cluster-based feature compression and cluster-based pruning, to calibrate the semantic 
distribution and reduce the redundancy of the datastore. We provide instructions on how to use them with kNN-models 
on this page, taking the IT domain corpus from 
[multi-domain parallel data](https://github.com/roeeaharoni/unsupervised-domain-clusters) as an example.



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
Generate the datastore with the pre-trained NMT model. Note that we set the environment variable 
`PCKMT_DATASTORE` in the script below to generate n-gram phrases and their corresponding probabilities, 
which are required by the cluster-based pruning strategy.
``` bash
export PCKMT_DATASTORE="1"

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


## Train compact network

In the script below, `vocab_size` denotes the number of words/subwords in the vocabulary 
of the pre-trained NMT model, `reduced_keys_dimension` denotes the feature dimension 
of the datastore keys after applying dimension reduction, which is also the output dimension 
of the last layer in the compact network. We set `reduced_keys_dimension` as 64 to follow 
the default setting of the official implementation. The trained compact network will be saved 
to the directory specified by `transformed_datastore`.

``` bash
domain="it"

data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`

reduced_keys_dimension=64
transformed_datastore=/path/to/transformed-datastore/${domain}
vocab_size=42024

mkdir -p ${transformed_datastore}

reduce_datastore_dims \
    --method PCKMT \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --keys-dtype fp16 \
    --transformed-datastore ${transformed_datastore} \
    --reduced-keys-dimension ${reduced_keys_dimension} \
    --stage train_pckmt \
    --vocab-size ${vocab_size} \
    --max-epoch 1000000000 \
    --max-update 70000 \
    --keep-best-checkpoints 10 \
    --betas "(0.9, 0.98)" \
    --log-interval 10 \
    --clip-norm 1.0
```


## Feature dimension reduction
The trained compact network is used to obtain separable semantic clusters and 
reduce the feature dimension of the datastore keys. `checkpoint_name` is the file name 
of the trained compact network saved in `transformed_datastore`. Additionally, the 
datastore after applying dimension reduction will be saved to `transformed_datastore`.

``` bash
domain="it"

data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
transformed_datastore=/path/to/transformed-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint_name="the name of selected checkpoint"
reduced_keys_dimension=64


reduce_datastore_dims \
    --method PCKMT \
    --stage apply_pckmt \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --transformed-datastore ${transformed_datastore} \
    --reduced-keys-dimension ${reduced_keys_dimension} \
    --checkpoint-name ${checkpoint_name}

# as the datastore values, n-gram phrases and their corresponding probabilities
# remain unchanged in this step, create a soft link between them to avoid copy
ln -s ${datastore}/values.npy ${transformed_datastore}/values.npy
ln -s ${datastore}/4_gram_values.npy ${transformed_datastore}/4_gram_values.npy
ln -s ${datastore}/4_gram_values_probs.npy ${transformed_datastore}/4_gram_values_probs.npy
```


## Prune datastore (optional)
The script applies cluster-based pruning strategy to the datastore. `sample_rate` denotes the proportion 
of entries to keep in each n-gram phrase group clustered by translation cost. It should be in the range 
of (0, 1). The pruned datastore will be saved to `pruned_datastore`. Note that this step is optional, 
you can skip it and move to the next step.

``` bash
domain="it"

transformed_datastore=/path/to/transformed-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
reduced_keys_dimension=64
pruned_datastore=/path/to/pruned-datastore/${domain}
sample_rate=0.9

mkdir -p ${pruned_datastore}

prune_datastore \
    --method cluster_based_pruning \
    --datastore ${transformed_datastore} \
    --datastore-size ${datastore_size} \
    --sample-rate ${sample_rate} \
    --keys-dimension ${reduced_keys_dimension} \
    --pruned-datastore ${pruned_datastore}
```


## Build the Faiss index
The script below will construct `IndexIVFPQ` with Faiss and save it 
as a file named `faiss.index` to the `pruned_datastore` directory.

``` bash
domain="it"

pruned_datastore=/path/to/pruned-datastore/${domain}
datastore_size="datastore size after pruning"
reduced_keys_dimension=64

build_faiss_index \
    --use-gpu \
    --datastore ${pruned_datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension ${reduced_keys_dimension} \
    --keys-dtype fp16 \
    --knn-fp16
```


## Train meta-k network
Train meta-k network on the validation set:
``` bash
domain="it"

num_neighbors="selected num_neighbors"
temperature="selected temperature"

knn_models=/path/to/knn_models
data_bin=/path/to/multi-domin-data-bin/${domain}
pruned_datastore=/path/to/pruned-datastore/${domain}
datastore_size="datastore size after pruning"
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
save_dir=/path/to/trained_model/${domain}

mkdir -p ${save_dir}

fairseq-train ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_adaptive_knn \
    --source-lang de \
    --target-lang en \
    --arch transformer_wmt_en_de_big \
    --dropout 0.2 \
    --encoder-ffn-embed-dim 8192 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --finetune-from-model ${checkpoint} \
    --validate-interval-updates 100 \
    --save-interval-updates 100 \
    --keep-interval-updates 1 \
    --max-update 5000 \
    --validate-after-updates 1000 \
    --save-interval 10000 \
    --validate-interval 100 \
    --keep-best-checkpoints 1 \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --train-subset valid \
    --valid-subset valid \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.001 \
    --batch-size 32 \
    --update-freq 1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-08 \
    --stop-min-lr 3e-05 \
    --lr 0.0003 \
    --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau \
    --lr-patience 5 \
    --lr-shrink 0.5 \
    --patience 30 \
    --max-epoch 500 \
    --datastore ${pruned_datastore} \
    --datastore-size ${datastore_size} \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --temperature-value ${temperature} \
    --save-dir ${save_dir}
```

## Evaluation
Translate the test set with Adaptive kNN-MT:
``` bash
domain="it"

num_neighbors="selected num_neighbors"
temperature="selected temperature"

knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
pruned_datastore=/path/to/pruned-datastore/${domain}
datastore_size="datastore size after pruning"
checkpoint=/path/to/trained_model/${domain}/checkpoint_best.pt

max_tokens=8000

fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_adaptive_knn \
    --datastore ${pruned_datastore} \
    --datastore-size ${datastore_size} \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
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
