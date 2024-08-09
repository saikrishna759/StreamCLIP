**StreamClip:** Intelligent Video Search and Genre-Based Extraction


## Organization
1. app.py: The main file where users can enter search queries and specify a genre for video content extraction through a Streamlit app interface.
2. resnet18_distilbert.py: A file that defines our first custom model. It uses Resnet18 for the image encoder and DistilBert for the text encoder.
3. vit_distilbert.py: A file that defines our second custom model. It uses VIT for the image encoder and DistilBert for the text encoder.
4. notebooks/clip.ipynb: A Jupyter notebook that employs CLIP for video content search.
5. notebooks/resnet18_distilbert.ipynb: A Jupyter notebook that employs our first custom model for video content search. It also downloads the 2014 COCO Image Caption Dataset, splits the data according to the Karpathy split, defines the custom model architecture, loss function, evalations, and tests.
6. notebooks/vit_distilbert.ipynb: A Jupyter notebook that employs our VIT custom model for video content search. It also downloads the 2014 COCO Image Caption Dataset, splits the data according to the Karpathy split, defines the custom model architecture, loss function, evalations, and tests.


## Installation
```
pip install streamlit
```
For more information, please refer to https://docs.streamlit.io/get-started/installation

## Run App

```
streamlit run app.py
```
For more information, please refer to https://docs.streamlit.io/develop/concepts/architecture/run-your-app

## Notebooks
All notebooks are runnable on Google Colab
