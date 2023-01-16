# Simple and Scalable Nearest Neighbor Machine Translation

[https://openreview.net/pdf?id=uu1GBD9SlLe](https://openreview.net/pdf?id=uu1GBD9SlLe)


**As the paper is still under review at ICLR 2023, to obey the double blind review policy
we hereby declare that we are not the author of the paper and do not know about the author identity.**


The paper [Simple and Scalable Nearest Neighbor Machine Translation](https://openreview.net/pdf?id=uu1GBD9SlLe) 
introduces SK-MT to reduce the storage requirement of kNN-MT and its variants which usually rely on the 
similarity search of dense vectors to perform retrieval across the whole datastore. Specifically, SK-MT 
first retrieves a small number of bilingual sentence pairs for each input sentence using the text retrieval 
algorithms such as BM25, then the retrieved sentence pairs are used to construct a tiny datastore on the fly.


This page includes instructions on how to use SK-MT with kNN-models, taking the IT domain corpus from 
[multi-domain parallel data](https://github.com/roeeaharoni/unsupervised-domain-clusters) as an example. 
**For efficiency, we do not use edit-distance to re-rank the search results of ElasticSearch as the time 
complexity of computing edit-distance is quadratic to the input string length, which is slightly different 
from the paper. Despite that, we found SK-MT can still achieve reasonably good performance without re-ranking.**


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


Binarize the validation and test sets with `fairseq-preprocess`:
``` bash
domain="it"
preprocessed_multi_domin_corpus=/path/to/preprocessed-multi-domin-corpus/${domain}
srcdict=/path/to/pretrained_model/dict.en.txt
data_bin=/path/to/multi-domin-data-bin/${domain}

fairseq-preprocess \
    --source-lang de \
    --target-lang en \
    --validpref ${preprocessed_multi_domin_corpus}/dev.bpe \
    --testpref ${preprocessed_multi_domin_corpus}/test.bpe \
    --srcdict ${srcdict} \
    --joined-dictionary \
    --workers 16 \
    --destdir ${data_bin}
```


## Add the training set to ElasticSearch
Add the training set to ElasticSearch to make it become searchable.
``` bash
domain="it"
preprocessed_multi_domin_corpus=/path/to/preprocessed-multi-domin-corpus/${domain}
elastic_password="please fill the password of the elastic user here"
index_name="please fill the name of index here"

es_knn_manager \
    --hosts https://localhost:9200 \
    --ca-certs /path/to/http_ca.crt \
    --elastic-password ${elastic_password} \
    --source-corpus-path ${preprocessed_multi_domin_corpus}/train.bpe.filtered.de \
    --target-corpus-path ${preprocessed_multi_domin_corpus}/train.bpe.filtered.en \
    --index-name ${index_name} \
    --operation build_datasetore
```


## Tune hyperparameters
The script below will retrieve 16 sentence pairs for each input sentence (specified by the `--size` flag ) 
and performs grid search over `num_neighbors` ∈ {1, 2, 4, 8, 16, 32, 64}, and `temperature_value` 
∈ {5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110} on the validation set.

``` bash
domain="it"
knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
elastic_password="please fill the password of the elastic user here"
index_name="please fill the name of index here"

max_tokens=8000

checkpoint=${project_dir}/data/wmt19.de-en/pretrained_models/wmt19.de-en.ffn8192.pt

for num_neighbors in 1 2 4 8 16 32 64; do
    for temperature_value in 5 10 20 30 40 50 60 70 80 90 100 110; do
        fairseq-generate ${data_bin} \
            --user-dir ${knn_models} \
            --task translation_es_knn \
            --hosts https://localhost:9200 \
            --ca-certs /path/to/http_ca.crt \
            --elastic-password ${elastic_password} \
            --index-name ${index_name} \
            --size 16 \
            --num-neighbors ${num_neighbors} \
            --temperature-value ${temperature_value} \
            --source-lang de \
            --target-lang en \
            --gen-subset valid \
            --path ${checkpoint} \
            --max-tokens ${max_tokens} \
            --beam 5 \
            --tokenizer moses \
            --post-process subword_nmt > tmp 2>&1

        cat tmp | grep -P "^D" | sort -V | cut -f 3- > tmp.final
        echo "num_neighbors: ${num_neighbors}, temperature_value: ${temperature_value}"
        sacrebleu -w 6 ${multi_domin_corpus}/${domain}/dev.${tgt_lang} --input tmp.final

    done
done
```


## Evaluation

Translate the test set with SK-MT:

``` bash
domain="it"

num_neighbors="selected num_neighbors"
temperature="selected temperature"

knn_models=/path/to/knn_models
multi_domin_corpus=/path/to/multi-domin-corpus
data_bin=/path/to/multi-domin-data-bin/${domain}
checkpoint=/path/to/pretrained_model/wmt19.de-en.ffn8192.pt
elastic_password="please fill the password of the elastic user here"
index_name="please fill the name of index here"

max_tokens=8000

fairseq-generate ${data_bin} \
    --user-dir ${knn_models} \
    --task translation_es_knn \
    --hosts https://localhost:9200 \
    --ca-certs /path/to/http_ca.crt \
    --elastic-password ${elastic_password} \
    --index-name ${index_name} \
    --size 16 \
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
