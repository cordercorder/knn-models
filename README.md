# kNN-models

<p align="center">
    <a href="#implemented-papers">Implemented Papers</a> • 
    <a href="#requirements-and-installation">Requirements and Installation</a> • 
    <a href="#getting-started">Getting Started</a> • 
    <a href="#benchmarks">Benchmarks</a> • 
    <a href="#acknowledgements">Acknowledgements</a>
</p>

## What's New
- 2022/10/04 kNN-models is publicly available

## Overview
kNN-models is a *k*-nearest neighbor augmented sequence modeling toolkit implemented based on [Fairseq](https://github.com/facebookresearch/fairseq). 
It enhances the pre-trained neural sequence to sequence model by retrieving from the external memory without expensive retraining. 

Main features:
 - Fast and memory efficient (please see [benchmarks](#benchmarks) for details)
 - Provide reference implementation of various *k*-nearest neighbor augmented sequence modeling papers (please see [Implemented-papers](#implemented-papers) for details)
 - Compatible with most of the pre-trained models in [Fairseq](https://github.com/facebookresearch/fairseq) (although only the transformer model has been well tested yet, we plan to conduct experiments with other models in the future)
 - Support similarity search with [Faiss](https://github.com/facebookresearch/faiss) and [Elasticsearch](https://github.com/elastic/elasticsearch-py) (retrieving with [Elasticsearch](https://github.com/elastic/elasticsearch-py) is an upcoming feature, it is still underdeveloped at the [es branch](https://github.com/cordercorder/knn-models/tree/es) and will merge into the main branch in the foreseeable future)
 - The [Faiss](https://github.com/facebookresearch/faiss) index can be placed on a GPU that is different from the one occupied by the model and sharded between multiple GPUs to avoid out of memory
 - The module which produces the intermediate hidden state to serve as datastore keys can be configured through command line arguments to adapt to the user's needs (it is the last layer in the decoder by default, please see the [BaseKnnConfig](https://github.com/cordercorder/knn-models/blob/main/knn_models/dataclass/knn_config.py#L78) for details)
 - Flexible configuration based on [Hydra](https://github.com/facebookresearch/hydra)


## Implemented Papers

The repository contains the reference implementation of following papers (sorted by publication date):
 - [Efficient Cluster-Based k-Nearest-Neighbor Machine Translation (ACL 2022)](examples/PCMKT/README.md)
 - [Efficient Nearest Neighbor Language Models (EMNLP 2021)](examples/efficient-knnlm/README.md)
 - [Adaptive Nearest Neighbor Machine Translation (ACL 2021)](examples/adaptive-knn-mt/README.md)
 - [Nearest Neighbor Machine Translation (ICLR 2021)](examples/knnmt/README.md)
 - [Generalization through Memorization: Nearest Neighbor Language Models (ICLR 2020)](examples/knnlm/README.md)


The detailed READMEs about how to reproduce them with kNN-models can be found in the `examples` folder.


## Requirements and Installation

The repository is developed and tested on Python 3.10, PyTorch 1.10.0, Fairseq 0.12.1, and Faiss-gpu 1.7.2. 
We recommend users keep the versions of these packages the same as ours to alleviate the compatibility issues, 
even though other versions may also work.

To install kNN-models and develop locally:
``` bath
git clone https://github.com/cordercorder/knn-models
cd knn-models
pip install -e ./
```


Note that `pip install -e ./` will check the packages in the Python environment to resolve the dependencies specified 
in `requirements.txt`. However, [Faiss](https://github.com/facebookresearch/faiss) installed 
through `conda` can not be identified by `pip`, which will result in the redundant 
[Faiss](https://github.com/facebookresearch/faiss) installation from PIP source. If you are pretty sure that 
all the packages required by this repository are installed well, you can run `python setup.py develop` to install 
kNN-models instead.


## Getting Started

We try to make the implementation independent of the model architecture during developing this repository. Consequently, 
we extend the task in [Fairseq](https://github.com/facebookresearch/fairseq) with the ability to perform similarity search. 
**As the task can be combined with different model architectures, we can enhance various pre-trained models with the external 
memory without modifying the official code of [Fairseq](https://github.com/facebookresearch/fairseq).** For example, the 
[kNN-MT](https://openreview.net/pdf?id=7wCBOfJ8hJM) can be implemented with just a few lines of code like the following:

``` python
from functools import partial
from dataclasses import dataclass
from fairseq.tasks.translation import (
    TranslationTask,
    TranslationConfig,
)
from fairseq.tasks import register_task
from fairseq.dataclass import FairseqDataclass
from knn_models.dataclass import KnnConfig
from knn_models.hook_utils import ForwardHook
from knn_models.knn_utils import (
    KnnSearch,
    get_captured_module,
    get_normalized_probs,
)


@dataclass
class TranslationKnnConfig(TranslationConfig):
    """config for nearest neighbor machine translation"""
    knn_config: KnnConfig = KnnConfig()


@register_task("translation_knn", dataclass=TranslationKnnConfig)
class TranslationKnnTask(TranslationTask):
    """task for nearest neighbor machine translation"""
    def __init__(self, cfg: TranslationKnnConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.knn_search = KnnSearch(cfg.knn_config)
        self.forward_hook = ForwardHook()

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)

        assert hasattr(model, "decoder"), \
            "TranslationKnnTask only supports the model with decoder! " \
            f"There is no decoder in {model.__class__.__name__}."
        
        # collect outputs from the specified module in decoder as the datastore keys
        captured_module_name = self.cfg.knn_config.module_to_capture
        captured_module = get_captured_module(model.decoder, captured_module_name)
        captured_module.register_forward_hook(self.forward_hook.forward_hook_function)

        # rewrite `get_normalized_probs` function to support kNN augmented NMT
        model.get_normalized_probs = partial(get_normalized_probs, self, model)
        return model
```


## Benchmarks

We measured the generation speed and GPU memory consumption during inference to evaluate the 
performance of kNN-models. We conducted experiments on [kNN-MT](https://openreview.net/pdf?id=7wCBOfJ8hJM) 
and [Adaptive kNN-MT](https://aclanthology.org/2021.acl-short.47.pdf) considering that they 
are dominant approaches to enabling retrieval argumented MT.


Following the common practice, we used the [multi-domain dataset](https://github.com/roeeaharoni/unsupervised-domain-clusters) 
[(Koehn & Knowles, 2017)](https://aclanthology.org/W17-3204.pdf) which was re-split by 
[Aharoni & Goldberg (2020)](https://aclanthology.org/2020.acl-main.692.pdf) for experiments and 
the WMT’19 German-English news translation task winner model [(Ng et al., 2019)](https://aclanthology.org/W19-5333.pdf) 
was adopted as the pre-trained NMT model. For [kNN-MT](https://openreview.net/pdf?id=7wCBOfJ8hJM), 
we tuned the hyperparameters (`num_neighbors`, `lambda`, `temperature`) on the validation sets 
according to the BLEU score. The hyperparameters for [Adaptive kNN-MT](https://aclanthology.org/2021.acl-short.47.pdf) 
were inherited from [kNN-MT](https://openreview.net/pdf?id=7wCBOfJ8hJM) except for `lambda`, which can be 
inferred from the Meta-*k*-Network of [Adaptive kNN-MT](https://aclanthology.org/2021.acl-short.47.pdf). We employed 
beam search with a beam size of 5 and a length penalty of 1.0 during decoding. **It is worth 
noting that only one GPU was used throughout the benchmark experiments and the 
[Faiss](https://github.com/facebookresearch/faiss) index was placed on GPU to speed up the search operation.**


The datastore size and the hyperparameters for each domain are presented below:

| | Medical | Law | IT | Koran | Subtitles |
| :----: | :----: | :----: | :----: | :----: | :----: |
| datastore size | 6501418 | 18857646 | 3449918 | 519897 |6209620 | 179484699 |
| num_neighbors | 8 | 8 | 16 | 16 | 16 | 16 |
| lambda | 0.7 | 0.7 | 0.6 | 0.7 | 0.5 | 0.6 |
| temperature | 5 | 5 | 5 | 20 | 20 | 10 |


The BLEU score of the pre-trained NMT model (Base MT), [kNN-MT](https://openreview.net/pdf?id=7wCBOfJ8hJM), and
[Adaptive kNN-MT](https://aclanthology.org/2021.acl-short.47.pdf) on the test sets for each domain 
are presented below:

|  | Medical | Law | IT | Koran | Subtitles |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Base MT | 41.87 | 45.96 | 38.52 | 17.07 | 29.39 |
| kNN-MT | 57.08 | 62.48 | 47.1 | 22.54 | 30.55 |
| Adaptive kNN-MT | 58.17 | 63.32 | 48.33 | 22.03 | 30.45|


### Generation Speed
As the generation speed usually varies between different runs and is highly dependent on 
the hardware environment, we performed each experiment 5 times and reported the mean and 
standard deviation of the generation speed on two different servers respectively.


The generation speed (token/s) of kNN-models on a server with 8 NVIDIA Tesla P100 GPUs (16GB), 
2 Intel Xeon Gold 6240 CPUs, and 256 GB of RAM is presented below (as there are sentences with 
more than 400 tokens in the test sets of medical and law domains, the generation speed is not 
available in the case of batch size set to 400):
<table>
    <tr>
        <th>Batch Size</th>
        <th> </th>
        <th>Medical</th>
        <th>Law</th>
        <th>IT</th>
        <th>Koran</th>
        <th>Subtitles</th>
    </tr>
    <tr>
        <td rowspan="3">400 tokens</td>
        <td>Base MT</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>593.67±12.92</td>
        <td>577.60±14.76</td>
        <td>1005.69±44.67</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>492.66±21.24</td>
        <td>488.79±20.47</td>
        <td>858.08±29.71</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>470.20±20.02</td>
        <td>455.39±16.95</td>
        <td>806.94±24.71</td>
    </tr>
    <tr>
        <td rowspan="3">800 tokens</td>
        <td>Base MT</td>
        <td>761.39±29.74</td>
        <td>705.84±7.99</td>
        <td>869.02±36.63</td>
        <td>830.49±34.10</td>
        <td>1502.55±29.31</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>625.08±24.04</td>
        <td>542.48±21.85</td>
        <td>738.49±31.51</td>
        <td>689.17±36.21</td>
        <td>1240.48±21.99</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>591.90±16.39</td>
        <td>521.86±12.26</td>
        <td>710.79±17.69</td>
        <td>642.82±20.04</td>
        <td>1190.69±15.46</td>
    </tr>
    <tr>
        <td rowspan="3">1600 tokens</td>
        <td>Base MT</td>
        <td>1033.93±30.34</td>
        <td>1000.80±34.31</td>
        <td>1195.03±41.52</td>
        <td>1138.84±41.03</td>
        <td>1859.79±10.62</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>829.28±22.33</td>
        <td>743.36±23.23</td>
        <td>993.22±22.14</td>
        <td>960.69±27.82</td>
        <td>1467.16±4.67</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>812.92±13.07</td>
        <td>715.14±18.86</td>
        <td>924.22±22.44</td>
        <td>903.87±16.43</td>
        <td>1408.14±16.42</td>
    </tr>
    <tr>
        <td rowspan="3">3200 tokens</td>
        <td>Base MT</td>
        <td>1335.80±20.57</td>
        <td>1294.52±15.47</td>
        <td>1445.16±20.55</td>
        <td>1497.09±16.30</td>
        <td>2047.57±19.40</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>1046.16±16.05</td>
        <td>940.59±9.40</td>
        <td>1197.04±18.48</td>
        <td>1247.45±17.36</td>
        <td>1586.45±10.99</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>1036.07±3.97</td>
        <td>917.63±10.08</td>
        <td>1189.73±5.70</td>
        <td>1203.48±9.22</td>
        <td>1577.00±12.18</td>
    </tr>
    <tr>
        <td rowspan="3">6400 tokens</td>
        <td>Base MT</td>
        <td>1563.36±11.48</td>
        <td>1522.87±11.01</td>
        <td>1613.63±17.39</td>
        <td>1716.00±11.16</td>
        <td>2126.56±19.66</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>1226.55±3.98</td>
        <td>1072.35±5.72</td>
        <td>1323.60±14.69</td>
        <td>1447.19±13.10</td>
        <td>1660.31±15.97</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>1193.37±13.58</td>
        <td>1043.77±6.62</td>
        <td>1293.78±11.54</td>
        <td>1408.91±7.27</td>
        <td>1648.06±17.63</td>
    </tr>
    <tr>
        <td rowspan="3">12800 tokens</td>
        <td>Base MT</td>
        <td>1675.49±9.45</td>
        <td>1633.76±9.67</td>
        <td>1647.95±12.20</td>
        <td>1803.01±10.18</td>
        <td>2197.24±13.67</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>1300.68±6.27</td>
        <td>1140.59±3.88</td>
        <td>1334.90±2.23</td>
        <td>1532.65±8.40</td>
        <td>1694.99±7.50</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>1275.62±10.28</td>
        <td>1125.35±5.66</td>
        <td>1323.47±9.31</td>
        <td>1500.19±10.48</td>
        <td>1699.80±10.55</td>
    </tr>
</table>


The generation speed (token/s) of kNN-models on a server with 8 NVIDIA GeForce GTX TITAN GPUs (24GB), 
2 Intel Xeon E5-2680 CPUs, and 256 GB of RAM is presented below:
<table>
    <tr>
        <th>Batch Size</th>
        <th> </th>
        <th>Medical</th>
        <th>Law</th>
        <th>IT</th>
        <th>Koran</th>
        <th>Subtitles</th>
    </tr>
    <tr>
        <td rowspan="3">400 tokens</td>
        <td>Base MT</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>435.83±15.51</td>
        <td>432.85±16.09</td>
        <td>844.25±57.33</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>408.02±21.15</td>
        <td>403.94±16.99</td>
        <td>759.71±51.01</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>393.35±25.35</td>
        <td>371.31±29.31</td>
        <td>724.04±42.07</td>
    </tr>
    <tr>
        <td rowspan="3">800 tokens</td>
        <td>Base MT</td>
        <td>634.81±15.64</td>
        <td>588.01±14.00</td>
        <td>743.54±42.92</td>
        <td>682.80±19.63</td>
        <td>1507.27±54.44</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>542.13±11.21</td>
        <td>481.48±8.66</td>
        <td>651.12±31.04</td>
        <td>618.70±11.19</td>
        <td>1261.36±44.09</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>526.43±33.34</td>
        <td>436.25±21.67</td>
        <td>633.04±29.44</td>
        <td>556.48±35.99</td>
        <td>1244.21±69.26</td>
    </tr>
    <tr>
        <td rowspan="3">1600 tokens</td>
        <td>Base MT</td>
        <td>967.79±14.60</td>
        <td>983.15±9.54</td>
        <td>1110.93±25.45</td>
        <td>1088.76±41.47</td>
        <td>2182.40±74.34</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>761.56±33.66</td>
        <td>726.35±25.67</td>
        <td>1040.71±17.07</td>
        <td>919.17±31.14</td>
        <td>1664.39±55.27</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>745.29±21.61</td>
        <td>719.38±27.49</td>
        <td>969.04±46.21</td>
        <td>915.46±52.70</td>
        <td>1601.80±38.00</td>
    </tr>
    <tr>
        <td rowspan="3">3200 tokens</td>
        <td>Base MT</td>
        <td>1526.37±43.21</td>
        <td>1488.71±78.56</td>
        <td>1665.54±66.93</td>
        <td>1885.99±13.26</td>
        <td>2645.62±80.18</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>1168.07±20.86</td>
        <td>1051.21±30.82</td>
        <td>1395.36±63.48</td>
        <td>1547.67±60.08</td>
        <td>2040.28±29.90</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>1135.30±63.46</td>
        <td>1037.96±54.62</td>
        <td>1335.45±60.56</td>
        <td>1442.43±52.53</td>
        <td>2032.88±47.17</td>
    </tr>
    <tr>
        <td rowspan="3">6400 tokens</td>
        <td>Base MT</td>
        <td>2078.05±14.57</td>
        <td>2038.81±60.04</td>
        <td>2078.64±55.91</td>
        <td>2397.98±11.12</td>
        <td>2838.64±12.76</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>1541.41±31.89</td>
        <td>1337.22±5.74</td>
        <td>1698.17±46.67</td>
        <td>1965.55±43.59</td>
        <td>2176.18±26.11</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>1494.57±22.87</td>
        <td>1326.34±24.34</td>
        <td>1695.56±42.75</td>
        <td>1902.53±45.91</td>
        <td>2173.67±25.10</td>
    </tr>
    <tr>
        <td rowspan="3">12800 tokens</td>
        <td>Base MT</td>
        <td>2377.90±20.36</td>
        <td>2374.11±6.77</td>
        <td>2158.86±21.50</td>
        <td>2589.23±40.78</td>
        <td>2986.30±31.20</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>1752.04±11.44</td>
        <td>1493.63±5.76</td>
        <td>1772.20±51.73</td>
        <td>2175.42±40.24</td>
        <td>2314.58±6.86</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>1719.02±36.40</td>
        <td>1476.38±13.23</td>
        <td>1765.07±47.39</td>
        <td>2117.49±45.74</td>
        <td>2313.21±44.98</td>
    </tr>
</table>


### GPU Memory Consumption

It is nontrivial to accurately measure the minimum amount of GPU memory to support model inference 
due to the complicated GPU memory management of [PyTorch](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) 
and [Faiss](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#scratch-memory). Nevertheless, 
to report the approximate minimum GPU memory requirement for inference, we disabled the memory caching 
of PyTorch by setting the value of the environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING` to `1` 
and monitored the maximum amount of used GPU memory every 10 milliseconds. 
We set the batch size to 12000 tokens to follow the [default setting of Fairseq](https://github.com/facebookresearch/fairseq/blob/main/fairseq_cli/generate.py#L72) for experiments.

The observed maximum GPU memory consumption of kNN-models during inference is presented below:
<table>
    <tr>
        <th>Batch Size</th>
        <th> </th>
        <th>Medical</th>
        <th>Law</th>
        <th>IT</th>
        <th>Koran</th>
        <th>Subtitles</th>
    </tr>
    <tr>
        <td rowspan="3">12000 tokens</td>
        <td>Base MT</td>
        <td>6363 MB</td>
        <td>6519 MB</td>
        <td>6509 MB</td>
        <td>6575 MB</td>
        <td>6349 MB</td>
    </tr>
    <tr>
        <td>kNN-MT</td>
        <td>8391 MB</td>
        <td>9383 MB</td>
        <td>8255 MB</td>
        <td>8155 MB</td>
        <td>8367 MB</td>
    </tr>
    <tr>
        <td>Adaptive kNN-MT</td>
        <td>8379 MB</td>
        <td>9403 MB</td>
        <td>8265 MB</td>
        <td>8153 MB</td>
        <td>8375 MB</td>
    </tr>
</table>


## Acknowledgements

We are extremely grateful to the research communities for their incredible work 
on retrieval argumented sequence modeling. This repository would not have been 
possible without them. Furthermore, we would also like to thank [wls](https://github.com/wonderseen)
for his generous help and valuable suggestions in replicating the [PCKMT](https://github.com/wonderseen/PCKMT). 
