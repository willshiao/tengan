TenGAN
-----

## Tensor Model
### How to run
Run the model using `Tensor_GAN.py`:

```bash
python Tensor_GAN.py --dataset=football_small --n_epochs=5000 --rank=100 --version=1.0.0 --model_name=${MODEL_NAME_HERE} --gen_class=NewCPTensorGenerator --disc_class=NewLayeredMultiviewDiscriminator --sampling_method=random_walk
```

It uses [Weights & Biases](https://wandb.ai/) by default for logging. You can make a free account there or just log locally.

### Installing
Install the requirements from `requirements.txt` using `pip install -r requirements.txt`. Unfortunately some requirements may be missing from the list (and may have to be installed as you go).

You will also have to install the `gutils` module from the **local** directory here. You can do this with the following commands:

```bash
$ cd gutils
$ python setup.py install
```

### Datasets
We used 4 datasets in the paper. 3 of them are publicly available:

- **football**: Can be downloaded [here](http://mlg.ucd.ie/aggregation/). Place it in `data/football`.
- **enron**: Can be downloaded [here](http://frostt.io/tensors/enron/). Extract and place it in `data/enron/enron.tns`. Then, you can run `scripts/sample_enron.py` to subsample from it.
- **nell2**: Can be downloaded [here](http://frostt.io/tensors/nell-2/). Extract and place it in `data/nell2/nell2.tns`. Then, you can run `scripts/sample_dense_nell.py` to subsample from it. 

### Key parameters
- `--rank`: controls the parameter `r` in  the model (e.g. generated matrix `A` will be `n`x`r`).
- `--dataset`: controls which dataset we use. Must correspond to the correct name in the create_tensor function of `dsloader/tensor_creator.py`.
- `--tensor_slices`: the number of views/slices in our graph/tensor. Depends on the dataset.

The other parameters can be seen in the `opt` dictionary in `Tensor_GAN.py`. Some may not be relevant to tensor generation and may be leftover artifacts from graph generation.

### Baseline

The HGEN included here is a modified version of [the original HGEN code](https://github.com/lingchen0331/HGEN/tree/fdb9f6801e7f0655a33565df83d0bb8e2a5d8fda).

`hgen_converter` converts a TenGAN dataset into a HGEN-compatible dataset. You can then run HGEN with the corresponding dataset name.

This also relies on the files in the `hin2vec` directory, which is a modified version of https://github.com/csiesheep/hin2vec with some bug fixes and updated to work with Python 3.

### Adding a new dataset
Add a new `elif` block to `create_tensor` and the relevant loading logic.

The function to load the dataset should return a `(np.array, list<nx.Graph>)` tuple (you may be able to get it to work with only one, but may also require a little modification). See `load_mtx` as an example.

Then, you can set the `dataset` and `tensor_slices` parameters when you run the model, and it should use the corresponding dataset. Note that you may have to append `_small` or `_large` to the dataset depending on if you want smaller or larger samples.
