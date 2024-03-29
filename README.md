# Attention-Based Neural Networks for Chroma Intra Prediction in Video Coding

| ![Marc Górriz][MarcGorriz-photo] | ![Saverio Blasi][SaverioBlasi-photo] | ![Alan F. Smeaton][AlanFmeaton-photo]  | ![Noel E. O’Connor][NoelEOConnor-photo] | ![Marta Mrak][MartaMrak-photo] |
|:-:|:-:|:-:|:-:|:-:|
| [Marc Górriz][MarcGorriz-web]  | [Saverio Blasi][SaverioBlasi-web] | [Alan F. Smeaton][AlanFmeaton-web] | [Noel E. O’Connor][NoelEOConnor-web] | [Marta Mrak][MartaMrak-web] |

[MarcGorriz-web]: https://www.bbc.co.uk/rd/people/marc-gorriz-blanch
[SaverioBlasi-web]: https://www.bbc.co.uk/rd/people/saverio-blasi
[MartaMrak-web]: https://www.bbc.co.uk/rd/people/marta-mrak
[AlanFmeaton-web]: https://www.insight-centre.org/users/alan-smeaton
[NoelEOConnor-web]: https://www.insight-centre.org/our-team/prof-noel-oconnor/

[MarcGorriz-photo]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/authors/MarcGorriz.jpg
[SaverioBlasi-photo]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/authors/SaverioBlasi.jpg
[MartaMrak-photo]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/authors/MartaMrak.jpg
[AlanFmeaton-photo]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/authors/AlanFSmeaton.jpg
[NoelEOConnor-photo]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/authors/NoelEOConnor.jpg

A joint collaboration between:

| ![logo-bbc] | ![logo-dcu] | ![logo-insight] |
|:-:|:-:|:-:|
| [BBC Research & Development][bbc-web] | [Dublin City University (DCU)][dcu-web] | [Insight Centre for Data Analytics][insight-web] |

[bbc-web]: https://www.bbc.co.uk/rd
[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/

[logo-bbc]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/logos/bbc.png  "BBC Research & Development"
[logo-insight]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/logos/dcu.png "Dublin City University"

## Abstract
Neural networks can be successfully used to improve several modules of advanced video coding schemes. In particular, compression of colour components was shown to greatly benefit from usage of machine learning models, thanks to the design of appropriate attention-based architectures that allow the prediction to exploit specific samples in the reference region. However, such architectures tend to be complex and computationally intense, and may be difficult to deploy in a practical video coding pipeline. This software implements the collection of simplifications presented in [this paper](https://github.com/bbc/intra-chroma-attentionCNN#publication) to reduce the complexity overhead of the attention-based architectures. The simplified models are integrated into the Versatile Video Coding (VVC) prediction pipeline, retaining compression efficiency of previous chroma intra-prediction methods based on neural networks, while offering different directions for significantly reducing coding complexity.

![visualisation-fig]

[visualisation-fig]: https://github.com/bbc/intra-chroma-attentionCNN/blob/main/logos/visualisation.png

## Publication
The software in this repository represents methods presented in "Attention-Based Neural Networks for Chroma Intra Prediction in Video Coding" which can be found at [IEEE Xplore](https://ieeexplore.ieee.org/document/9292660).

Please cite with the following Bibtex code:
```
@ARTICLE{9292660,
  author={M. {Gorrizblanch} and S. G. {Blasi} and A. {Smeaton} and N. {O'Connor} and M. {Mrak}},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Attention-Based Neural Networks for Chroma Intra Prediction in Video Coding}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSTSP.2020.3044482}}

```
## How to use

### Dependencies

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). Also, this code should be compatible with Python 3.6. Moreover, the proposed implementation in [VTM-7.0](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git) is written in C++11 as the original features.

### Prepare data

Training examples were extracted from the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/), which contains high-definition high-resolution content of large diversity. This database contains 800 training samples and 100 samples for validation, providing 6 lower resolution versions with downsampling by  factors of 2, 3 and 4 with a bilinear and unknown filters. For each data instance, one resolution was randomly selected and then M blocks of each NxN sizes (N=4, 8, 16) were chosen, making balanced sets between block sizes and uniformed spatial selections within each image.

Training and validation images are organised in 7 resolution classes. We expect the directory structure to be the following:
```
path/to/DIV2K/
  train/
    0/ # HR: 0001.png - 0800.png
    1/ # LR_bicubic_X2: 0001.png - 0800.png
    2/ # LR_unknown_X2: 0001.png - 0800.png
    3/ # LR_bicubic_X3: 0001.png - 0800.png
    4/ # LR_unknown_X3: 0001.png - 0800.png
    5/ # LR_bicubic_X4: 0001.png - 0800.png
    6/ # LR_unknown_X4: 0001.png - 0800.png
  val/
    0/ # HR: 0801.png - 0900.png
    1/ # LR_bicubic_X2: 0801.png - 0900.png
    2/ # LR_unknown_X2: 0801.png - 0900.png
    3/ # LR_bicubic_X3: 0801.png - 0900.png
    4/ # LR_unknown_X3: 0801.png - 0900.png
    5/ # LR_bicubic_X4: 0801.png - 0900.png
    6/ # LR_unknown_X4: 0801.png - 0900.png
```

To create random training and validation blocks of the desired resolution run:
```
python create_database.py -i path/to/DIV2K -o path/to/blocks
```

### Train a model configuration

To train a model run the ```train.py``` script selecting the desired configuration. Update the size-dependent configurations at ```config/att/``` and the multi-models at ```config/att_multi/```:
```
python train.py -c [path/to/cf_file].py -g [gpu number]
```

### Deploy a model scheme

In order to integrate the trained models into VTM 7.0, we need to export their parameters and apply the proposed simplifications. As explained in the paper, 3 multi-model schemes are considered, to deploy its parameters update the deployment config file at ```config/deploy/``` and run:
```
python deploy.py -c config/deploy/scheme[X].py
```

The resultant weights and bias will be stored in the deploy path defined in the config file. In order to integrate them into the codec follow the next section to compile the updated VTM-7.0 version and copy the deployed arrays in ```VVCSoftware_VTM/source/Lib/CommonLib/NNIntraChromaPrediction.h```.

### Update VTM-7.0 with the proposed schemes

In order to generate a VTM-7.0 updated version with the proposed schemes, clone the original version and apply the patch differences relative to each scheme located at ```VTM-7.0-schemes/scheme[X].patch```:
```
git clone -b VTM-7.0 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
git apply ../VTM-7.0-schemes/scheme[X].patch
```
To compile the generated VTM-7.0 version follow the official instructions in ```VVCSoftware_VTM/README.md```.

### Reproduce the results

All the schemes are evaluated against a constrained VTM-7.0 anchor, whereby the VVC partitioning process is limited to using only square blocks of 4, 8 and 16 pixels. In order to generate the constrained VTM-7.0 anchor in this paper, apply the patch difference located at ```VTM-7.0-schemes/square_anchor.patch```.

## Improvements: Spatial Information Refinement

We collaborated with Northwestern Polytechnical University (Xi’an, China) to improve the schemes proposed in this work. Two new schemes for spatial information refinement are proposed: adding a down-sampling branch and adding location maps. A down-sampling filter is learnt, in order to select the most suitable down-sampling luma features for chroma prediction. Moreover, in order to allow the network to predict pixels with different importance levels, the position information of the current block and the boundary information are used to construct a feature map, called location map, which further guides the prediction process. 

For more information, refer to the [pre-print paper](https://arxiv.org/abs/2109.11913) "Spatial Information Refinement for Chroma Intra Prediction in Video Coding", accepted for publication in APSIPA 2021. Moreover, an open-source implementation can be found in [this repository](https://github.com/Chengyi-Zou/intra-chroma-attentionCNN-refinement), where the proposed refinement schemes can be applied in top of VTM by means of the corresponding patch differences. 

## Acknowledgements
This work has been conducted within the project
JOLT. This project is funded by the European Union’s Horizon 2020 research
and innovation programme under the Marie Skłodowska Curie grant agreement No 765140.

| ![JOLT-photo] | ![EU-photo] |
|:-:|:-:|
| [JOLT Project](JOLT-web) | [European Comission](EU-web) |


[JOLT-photo]: https://github.com/bbc/ColorGAN/blob/master/logos/jolt.png "JOLT"
[EU-photo]: https://github.com/bbc/ColorGAN/blob/master/logos/eu.png "European Comission"


[JOLT-web]: http://joltetn.eu/
[EU-web]: https://ec.europa.eu/programmes/horizon2020/en

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/marc-gorriz/ColorGAN/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:marc.gorrizblanch@bbc.co.uk>.
