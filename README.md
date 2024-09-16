# Hypergraph Wavelets

## Conda environment
We recommend the usage of a conda environment for running the experiments.
The used conda environment can be found at ```environment.yml```
## Dataset download

The data can be downloaded on the following link: https://sea-ad-spatial-transcriptomics.s3.amazonaws.com/index.html#middle-temporal-gyrus/all_donors-h5ad/

You can then extract your zip file into the ```data/raw``` folder.

### Dataset pre-processing
After all the data was extracted into the raw files, you can use the notebook found at : ```notebooks/create_patients_section_files.ipynb``` to process the raw data and divide the dataset from patients level to section level data.
This notebook will create all the section data in the folder: ```data/interim/section_data```

## Extracting wavelets features

Once the data is on the correct location you can simple run ```python3 main.py```

The resulting wavelets features will be stored on ```data/processed/wavelet_features```

## Evaluating performance

We evaluated performance using vendi score and linear probing.
Each one of those can be computed on the following scripts:

```python3 vendi.py```

```python3 linear_probing.py```