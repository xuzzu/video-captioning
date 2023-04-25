# Semantics-Assisted Video Captioning Model Trained with Scheduled Sampling in PyTorch

This repository contains the implementation of a Semantics-Assisted Video Captioning Model Trained with Scheduled Sampling in PyTorch. The model is trained on the MSVD dataset and achieves a performance of about 0.7 for ROUGE_L. 
Overall working principle: ResNet based model (ResNeXt) and ShuffleNet extract 2D and 3D features from the video dataset. These features are fed into the LSTM to predict tokens.

## Project Structure

Some description of files:

- `\extraction_models`: Used for feature extraction from raw dataset
- `inference.py`: Runs video captioning using saved PyTorch checkpoints
- `scn.py`: Defines the Semantic Context Network used in the model
- `train.py`: Runs training
- `utils.py, dataset.py, helper.py`: Supportive modules

## Usage

1. Clone the repository to your local machine.
2. Install the dependencies listed in `requirements.yml` using Conda.
3. Download the MSVD dataset and extract the videos and captions to a desired directory.
4. Run `extract_features.py` to extract features from the videos using the pre-trained models.
5. Run `train.py` with desired parameters to train the model.
6. Run `inference.py` to run tests.

Note that the paths to the dataset, pre-trained models, and other important parameters are configured during the run by passing it as parameters. You can chech helper.py for more details.

## Acknowledgements

Some code and implementation details are taken from these sources:
- https://arxiv.org/abs/1909.00121
- https://pytorch.org/hub/pytorch_vision_vgg/
- https://openaccess.thecvf.com/content/ICCV2021W/VSPW/papers/Wang_LiteEdge_Lightweight_Semantic_Edge_Detection_Network_ICCVW_2021_paper.pdf
- https://github.com/okankop/Efficient-3DCNNs
- https://aclanthology.org/D15-1199.pdf
