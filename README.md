
# Rapid Facial Reconstruction (RFR)

<!-- About the Project -->
## About the Project

This project involves integrating a tuned Wav2Vec2 speech-to-text model with a tuned Stable Diffusion v1.5 text-to-image model to produce a speech-to-text-to image pipeline for rapid facial reconstruction in forensic settings. Wav2Vec2 was tuned using the "dev-clean" subset of LibriSpeech ASR, a dataset of 1,000+ hours of English audiobook recordings. Stable Diffusion v1.5 was tuned using a subset of FaceCaption-15M, a dataset of 15 million faces and corresponding textual descriptions. More information can be found within the accompanying report.

See the [project demo](https://youtu.be/DYbUo5yk29Y).

<!-- Getting Started -->
## Getting Started

The Wav2Vec2 model was fine-tuned in Kaggle using a P100 GPU, while the Stable Diffusion v1.5 model was fine-tuned in Google Colab using an A100 GPU. The code within this repository is identical to the code ran on Kaggle and Colab.

### Running Locally

1. Clone the repository:
    
    ```git clone https://github.com/joshual55/rapid-facial-reconstruction.git```

2. Create the conda environment using the provided YML file:

    ```conda env create -f environment.yml```
    
3. Download the [LibriSpeech dataset](https://www.openslr.org/12): 
    - Download the ```dev-clean.tar.gz``` and ```test-clean.tar.gz``` subsets
    - Place the subsets in ```wav2vec2/```

4. Download the preprocessed FaceCaption-15M [train](https://drive.google.com/drive/folders/1d5lVzygHI7pkWo7dm-JQ2M5yanuGNVda?usp=sharing) and [test](https://drive.google.com/drive/folders/1hgcE79cba6-OTv-gq_3X8xPhRbMsWA2w?usp=sharing) data, and the [final diffusion model](https://drive.google.com/drive/folders/1BtmN4hGjbb-PRCHP_xdQpzvUjdS9IJGt?usp=sharing):
	- Place all in ```stable_diffusion_v15/```

5. Adjust necessary code snippets:
	- Change file paths throughout:
		-  ```wav2vec2/main.ipynb```
		- ```stable_diffusion_v15/main.ipynb```
		- ```gradio.ipynb```
	- Remove Google Colab drive import and mounting from:
		- ```stable_diffusion_v15/main.ipynb```
		- ```gradio.ipynb```

6. Run the code:
	- ```wav2vec2/main.ipynb```
		- All implementation for the Wav2Vec2 model
		- _Model testing can run standalone by loading the Datasets, DataLoaders, and trained model beforehand_
	- ```stable_diffusion_v15/main.ipynb```
		- All implementation for the Stable Diffusion v1.5 model
		- _Model testing can run standalone by loading the Datasets, DataLoaders, and trained model beforehand_
	-  ```gradio.ipynb```
		- Live Gradio testing interface

<!-- Files -->
## Project Files

### ```gradio.ipynb```

- The live Gradio testing interface for the project
- Provides a clean GUI for a user to easily record an audio snippet and pass it into both models to generate an output image

### ```wav2vec2/```

#### ```wav2vec2/main.ipynb```

- The primary code file for the Wav2Vec2 model, implementing all loading, preprocessing, training, and testing logic
- The exact file that was used in Kaggle

#### ```wav2vec2/wav2vec2_model/*```

- The final, fine-tuned Wav2Vec model folder
- Contains ```config.json``` and ```model.safetensors``` files

#### ```wav2vec2/train_dataset.pth```

- The training Dataset object for Wav2vec2

#### ```wav2vec2/test_dataset.pth```

- The testing Dataset object for Wav2Vec2

### ```stable_diffusion_v15/```

#### ```stable_diffusion_v15/main.ipynb```

- The primary code file for the Stable Diffusion v1.5 model, implementing all loading, preprocessing, training, and testing logic
- The exact file that was used in Google Colab

#### ```stable_diffusion_v15/train_dataset.pth```

- The training Dataset object for Stable Diffusion v1.5

#### ```stable_diffusion_v15/test_dataset.pth```

- The testing Dataset object for Stable Diffusion v1.5

### ```Project 3.ipynb```

- The Project 3 instructions file present when the repository was created

### ```environment.yml```

- The Conda environment used to run the code locally

### ```Report.pdf```

- The accompanying IEEE report for this project.

### ```README.md```

- The README currently being read.

<!-- Author(s) -->
## Author(s)

Joshua Lamb - [github.com/joshual55](https://github.com/joshual55) - joshua.lamb55@gmail.com

Project Link: https://github.com/joshual55/rapid-facial-reconstruction

<!-- Acknowledgements -->
## Acknowledgements

- [Hugging Face Wav2Vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)
- [Hugging Face Wav2Vec2 Tutorial](https://huggingface.co/docs/transformers/tasks/asr)
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)

<!-- Thank you -->
## Thank you
