#!/bin/bash

set -e

data_dir=${1}
output_dir=${2}


mosesdecoder=/path/to/mosesdecoder
fastbpe=/path/to/fastBPE
pretrained_model=/path/to/pretrained_model


scripts=${mosesdecoder}/scripts
tokenizer=${scripts}/tokenizer/tokenizer.perl
clean=${scripts}/training/clean-corpus-n.perl
norm_punc=${scripts}/tokenizer/normalize-punctuation.perl
remove_non_print_char=${scripts}/tokenizer/remove-non-printing-char.perl

bpecodes=${pretrained_model}/ende30k.fastbpe.code
vocab=${pretrained_model}/dict.en.txt

src=de
tgt=en

for lang in ${src} ${tgt}; do
    cat ${data_dir}/train.${lang} | \
        perl ${norm_punc} ${lang} | \
            perl ${remove_non_print_char} | \
                perl ${tokenizer} -threads 8 -a -l ${lang} > ${output_dir}/train.tok.${lang}
    
    ${fastbpe}/fast applybpe ${output_dir}/train.bpe.${lang} ${output_dir}/train.tok.${lang} ${bpecodes} ${vocab}
done

perl ${clean} -ratio 3.0 ${output_dir}/train.bpe ${src} ${tgt} ${output_dir}/train.bpe.filtered 1 250

for split in dev test; do
    cat ${data_dir}/${split}.${src} | \
        perl ${norm_punc} ${src} | \
            perl ${remove_non_print_char} | \
                perl ${tokenizer} -threads 8 -a -l ${src} > ${output_dir}/${split}.tok.${src}

    cat ${data_dir}/${split}.${tgt} | \
        perl ${tokenizer} -threads 8 -a -l ${tgt} > ${output_dir}/${split}.tok.${tgt}

    ${fastbpe}/fast applybpe ${output_dir}/${split}.bpe.${src} ${output_dir}/${split}.tok.${src} ${bpecodes} ${vocab}

    ${fastbpe}/fast applybpe ${output_dir}/${split}.bpe.${tgt} ${output_dir}/${split}.tok.${tgt} ${bpecodes} ${vocab}
done
