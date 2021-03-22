# Attention-Based Neural Networks for Chroma Intra Prediction in Video Coding

| ![Marc Górriz][MarcGorriz-photo] | ![Saverio Blasi][SaverioBlasi-photo] | ![Alan F. Smeaton][AlanFmeaton-photo]  | ![Noel E. O’Connor][NoelEOConnor-photo] | ![Marta Mrak][MartaMrak-photo] |
|:-:|:-:|:-:|:-:|:-:|
| [Marc Górriz][MarcGorriz-web]  | [Saverio Blasi][SaverioBlasi-web] | [Alan F. Smeaton][AlanFmeaton-web] | [Noel E. O’Connor][NoelEOConnor-web] | [Marta Mrak][MartaMrak-web] |

[MarcGorriz-web]: https://www.bbc.co.uk/rd/people/marc-gorriz-blanch
[SaverioBlasi-web]: https://www.bbc.co.uk/rd/people/saverio-blasi
[MartaMrak-web]: https://www.bbc.co.uk/rd/people/marta-mrak
[AlanFmeaton-web]: https://www.insight-centre.org/users/alan-smeaton
[NoelEOConnor-web]: https://www.insight-centre.org/our-team/prof-noel-oconnor/

[MarcGorriz-photo]: https://github.com/marc-gorriz/Intra-Chroma-AttentionCNN/blob/master/authors/MarcGorriz.jpg
[SaverioBlasi-photo]: https://github.com/marc-gorriz/Intra-Chroma-AttentionCNN/blob/master/authors/SaverioBlasi.jpg
[MartaMrak-photo]: https://github.com/marc-gorriz/Intra-Chroma-AttentionCNN/blob/master/authors/MartaMrak.jpg
[AlanFmeaton-photo]: https://github.com/marc-gorriz/Intra-Chroma-AttentionCNN/blob/master/authors/AlanFSmeaton.jpg
[NoelEOConnor-photo]: https://github.com/marc-gorriz/Intra-Chroma-AttentionCNN/blob/master/authors/NoelEOConnor.jpg

A joint collaboration between:

| ![logo-bbc] | ![logo-dcu] | ![logo-insight] |
|:-:|:-:|:-:|
| [BBC Research & Development][bbc-web] | [Dublin City University (DCU)][dcu-web] | [Insight Centre for Data Analytics][insight-web] |

[bbc-web]: https://www.bbc.co.uk/rd
[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/

[logo-bbc]: https://github.com/bbc/ColorGAN/blob/master/logos/bbc.png  "BBC Research & Development"
[logo-insight]: https://github.com/bbc/ColorGAN/blob/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://github.com/bbc/ColorGAN/blob/master/logos/dcu.png "Dublin City University"

## Abstract
Neural networks can be successfully used to improve several modules of advanced video coding schemes. In particular, compression of colour components was shown to greatly benefit from usage of machine learning models, thanks to the design of appropriate attention-based architectures that allow the prediction to exploit specific samples in the reference region. However, such architectures tend to be complex and computationally intense, and may be difficult to deploy in a practical video coding pipeline. This work focuses on reducing the complexity of such methodologies, to design a set of simplified and cost-effective attention-based architectures for chroma intra-prediction. A novel size-agnostic multi-model approach is proposed to reduce the complexity of the inference process. The resulting simplified architecture is still capable of outperforming state-of-the-art methods. Moreover, a collection of simplifications is presented in this paper, to further reduce the complexity overhead of the proposed prediction architecture. Thanks to these simplifications, a reduction in the number of parameters of around 90% is achieved with respect to the original attention-based methodologies. Simplifications include a framework for reducing the overhead of the convolutional operations, a simplified cross-component processing model integrated into the original architecture, and a methodology to perform integer-precision approximations with the aim to obtain fast and hardware-aware implementations. The proposed schemes are integrated into the Versatile Video Coding (VVC) prediction pipeline, retaining compression efficiency of state-of-the-art chroma intra-prediction methods based on neural networks, while offering different directions for significantly reducing coding complexity.

![visualisation-fig]

[visualisation-fig]: https://github.com/marc-gorriz/Intra-Chroma-AttentionCNN/blob/master/logos/visualisation.png

## Publication
IEEE Journal of Selected Topics in Signal Processing. Find the paper discribing our work on [IEEE Xplore](https://ieeexplore.ieee.org/document/9292660).

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

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). Also, this code should be compatible with Python 3.6.

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

In order to integrate the trained models into VTM 7.0, we need to export their parameters and apply the proposed simplifications. As explained in the paper, 3 multi-model schemes are considered, to deploy its parameters update the deployment config file at ```config/att_multi/``` and run:
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
