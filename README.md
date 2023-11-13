Graph Convolutional Network with elastic topology (GCNet)
====
This is the implementation of GCNet in our manuscript:

Graph Convolutional Network with Elastic Topology, submitted to Pattern Recognition.

## Requirement

  * Python: 3.9.12
  * PyTorch: 1.11.0
  * Pytorch Geometric: 2.0.4
  * Numpy: 1.21.5
  * Scikit-learn: 1.1.0
  * Scipy: 1.8.0
  * Texttable: 1.6.4
  * Tqdm: 4.64.0

## Quick Start

```
python main.py --dataset Cora --model GCNet
```

## Datasets

  * Cora
  * Citeseer
  * ACM
  * BlogCatalog
  * Flickr
  * UAI

Please unzip ```./datasets/datasets.7z``` first
