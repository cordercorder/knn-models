# What Knowledge Is Needed? Towards Explainable Memory for kNN-MT Domain Adaptation

[https://arxiv.org/abs/2211.04052](https://arxiv.org/abs/2211.04052)


The paper [What Knowledge Is Needed? Towards Explainable Memory for kNN-MT Domain Adaptation](https://arxiv.org/abs/2211.04052) 
introduces PLAC (<u>P</u>runing with <u>L</u>oc<u>A</u>l <u>C</u>orrectness) to prune the datastore while retaining the translation performance of kNN-MT as much as possible. PLAC is built on a criterion named knowledge margin, which is used to measure the 
correctness of NMT model prediction. For PLAC, the entries in the datastore whose knowledge margin is higher than a manually 
defined threshold are considered pruning candidates. 


We provide instructions on how to use PLAC with kNN-models to prune datastore on this page, taking the IT domain corpus from 
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
`GENERATE_GREEDY_MT_PREDICTION` in the script below to save the token with the highest 
probability at each time step during teacher-forcing decoding, which is required by PLAC.
``` bash
export GENERATE_GREEDY_MT_PREDICTION="1"

domain="it"
knn_models=/path/to/knn_models
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt

mkdir -p ${datastore}

generate_mt_datastore ${data_bin} \
    --user-dir ${knn_models} \
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
The script below will construct the `IndexIVFPQ` index with Faiss and save it 
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


## Prune the datastore with PLAC
``` bash
domain="it"
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`

pruned_datastore=/path/to/pruned-datastore/${domain}

knowledge_margin="please fill in the knowledge margin here"
pruning_rate="please fill in the pruning_rate here"

mkdir -p ${pruned_datastore}

prune_datastore \
    --method plac_pruning \
    --plac-stage get_plac_pruning_candicate \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --pruned-datastore ${pruned_datastore} \
    --use-gpu \
    --knowledge-margin 4 \
    --knn-fp16


prune_datastore \
    --method plac_pruning \
    --plac-stage apply_plac_pruning \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --pruned-datastore ${pruned_datastore} \
    --pruning-rate ${pruning_rate}
```


## Build the Faiss index for the pruned datastore
``` bash
domain="it"
datastore=/path/to/pruned-datastore/${domain}
datastore_size="please fill in the size of the pruned datastore here"

build_faiss_index \
    --use-gpu \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --keys-dimension 1024 \
    --keys-dtype fp16 \
    --knn-fp16
```


## Train meta-k network
Train meta-k network with the validation set:
``` bash
domain="it"

num_neighbors="selected num_neighbors"
temperature="selected temperature"

knn_models=/path/to/knn_models
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/pruned-datastore/${domain}
datastore_size="please fill in the size of the pruned datastore here"
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
save_dir=/path/to/trained_model/${domain}

mkdir -p ${save_dir}

CUDA_VISIBLE_DEVICES=0 fairseq-train ${data_bin} \
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
    --datastore ${datastore} \
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
datastore=/path/to/pruned-datastore/${domain}
datastore_size="please fill in the size of the pruned datastore here"
checkpoint=/path/to/trained_model/${domain}/checkpoint_best.pt

max_tokens=8000

fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_adaptive_knn \
    --datastore ${datastore} \
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
