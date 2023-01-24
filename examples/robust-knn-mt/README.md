# Towards Robust k-Nearest-Neighbor Machine Translation

[https://arxiv.org/abs/2210.08808](https://arxiv.org/abs/2210.08808)


The paper [Towards Robust k-Nearest-Neighbor Machine Translation](https://arxiv.org/abs/2210.08808) 
proposes Distribution Calibration (DC) network and Weight Prediction (WP) network to leverage  
the model confidence to improve the final probability distribution. Additionally, two types of 
perturbations are injected into the retrieved items during training to improve the distribution 
further. We provide an implementation of this paper in kNN-models and show a usage example 
with a specific focus on the IT domain corpus of the
[multi-domain parallel data](https://github.com/roeeaharoni/unsupervised-domain-clusters)
on this page. 


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


## Train DC network and WP network
Train DC network WP network on the validation set:
``` bash
domain="it"

num_neighbors="selected num_neighbors"

knn_models=/path/to/knn_models
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
save_dir=/path/to/trained_model/${domain}

fairseq-train ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_robust_knn \
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
    --criterion label_smoothed_cross_entropy_for_robust_knn_mt \
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
    --load-keys \
    --move-to-memory \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --save-dir ${save_dir}
```

## Evaluation
Translate the test set with the trained model:
``` bash
domain="it"

num_neighbors="selected num_neighbors"

knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/trained_model/${domain}/checkpoint_best.pt

max_tokens=8000

fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_robust_knn \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --load-keys \
    --move-to-memory \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
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
