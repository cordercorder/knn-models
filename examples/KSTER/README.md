# Learning Kernel-Smoothed Machine Translation with Retrieved Examples

[https://aclanthology.org/2021.emnlp-main.579.pdf](https://aclanthology.org/2021.emnlp-main.579.pdf)


The paper introduces KSTER (Kernel-Smoothed Translation with Example Retrieval) to improve 
the generalization ability of kNN-MT when there is a distribution discrepancy between the 
datastore and input text. Moreover, the BLEU score can be further improved over vanilla kNN-MT 
even in the case that the datastore covers the domain of the input text.


We provide instructions on how to use KSTER with kNN-models on this page, taking the IT domain 
corpus from [multi-domain parallel data](https://github.com/roeeaharoni/unsupervised-domain-clusters) 
as an example. 


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


## Train KSTER
Train KSTER with the domain specific training set:
``` bash
domain="it"

# set the number of retrieved items to 16 
# to follow the description in the paper
num_neighbors=16

knn_models=/path/to/knn_models
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
save_dir=/path/to/trained_model/${domain}

mkdir -p ${save_dir}

nohup fairseq-train ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_kernel_smoothed_knn \
    --source-lang de \
    --target-lang en \
    --arch transformer_wmt_en_de_big \
    --dropout 0.2 \
    --encoder-ffn-embed-dim 8192 \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --finetune-from-model ${checkpoint} \
    --max-update 5000 \
    --max-epoch 500 \
    --max-tokens 4096 \
    --keep-best-checkpoints 5 \
    --save-interval-updates 500 \
    --no-save-optimizer-state \
    --no-epoch-checkpoints \
    --train-subset train \
    --valid-subset valid \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --update-freq 8 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --lr 0.0002 \
    --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau \
    --lr-patience 8 \
    --lr-shrink 0.7 \
    --datastore ${datastore} \
    --datastore-size ${datastore_size} \
    --load-keys \
    --move-to-memory \
    --knn-fp16 \
    --num-neighbors ${num_neighbors} \
    --save-dir ${save_dir}
```


## Select the best checkpoint
Select the best checkpoint according to the BLEU score on the validation set

``` bash
domain="it"

num_neighbors=16

knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`
save_dir=/path/to/trained_model/${domain}

checkpoint_name_array=(
    "please fill in the names of saved checkpoints here"
)

max_tokens=8000

for checkpoint_name in ${checkpoint_name_array[@]}; do
    fairseq-generate ${data_bin} \
        --user-dir ${knn_models} \
        --task translation_kernel_smoothed_knn \
        --datastore ${datastore} \
        --datastore-size ${datastore_size} \
        --load-keys \
        --move-to-memory \
        --knn-fp16 \
        --num-neighbors ${num_neighbors} \
        --source-lang de \
        --target-lang en \
        --gen-subset valid \
        --path ${save_dir}/${checkpoint_name} \
        --max-tokens ${max_tokens} \
        --beam 5 \
        --tokenizer moses \
        --post-process subword_nmt > raw_sys.de-en.en 2>&1

    cat raw_sys.de-en.en | grep -P "^D" | sort -V | cut -f 3- > sys.de-en.en
    score=`sacrebleu --score-only -w 6 ${multi_domin_corpus}/${domain}/dev.en --input sys.de-en.en`
    echo "checkpoint_name: ${checkpoint_name}, BLEU: ${score}"
done
```


## Evaluation
Translate the test set with the selected checkpoint:
``` bash
domain="it"

num_neighbors=16

knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
datastore=/path/to/multi-domin-datastore/${domain}
datastore_size=`count_tokens -d ${data_bin}/train.de-en.en`

checkpoint_name="please fill in the name of selected checkpoint here"
checkpoint=/path/to/trained_model/${domain}/${checkpoint_name}

max_tokens=8000

fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_kernel_smoothed_knn \
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
