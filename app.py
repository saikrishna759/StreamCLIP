import streamlit as st
import torch
import numpy as np
import yt_dlp
import cv2
from transformers import CLIPProcessor, CLIPModel, DistilBertTokenizerFast
from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import tempfile
import ffmpeg
from resnet18_distilbert import Model
from vit_distlbert import ViT_Model
import gdown
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

# Initialize the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_youtube_video(youtube_url):
    parts = url.split('=')
    video_id = parts[-1] if len(parts) > 1 else None
    output_path = f"downloaded_video_{video_id}.mp4"  # Dynamic file name
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path


def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        st.error("Failed to load video.")
        return None, None

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i % (fps // 2) == 0:  # Extract every 0.5 seconds
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    video.release()
    return frames, fps



def convert_video_to_h264(input_path, output_path):
    try:
        # Ensure the output directory exists
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Run ffmpeg conversion
        ffmpeg.input(input_path).output(output_path, vcodec='libx264').run(overwrite_output=True)
        st.success("Video converted successfully and is ready to be played.")
    except ffmpeg.Error as e:
        st.error(f"Failed to convert video due to FFmpeg error: {str(e.stderr)}")
    except Exception as e:
        st.error(f"An error occurred during video conversion: {str(e)}")


def frames_to_video(frames, start_indices, fps, output_path):
    if not start_indices:
        st.warning("No genre-specific clips found.")
        return None
    
    # Assuming each clip starts at start_indices and lasts for 5 seconds
    clips = [frames[start:start + int(fps * 1)] for start in start_indices]  

    if not clips:
        st.warning("No clips generated from the specified indices.")
        return None
    
    # Assuming the first frame is representative for all (in terms of dimensions)
    first_frame = np.array(frames[0])  # Ensure frames are NumPy arrays
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'DIVX' is used for AVI files, you can change this accordingly
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for clip in clips:
        for frame in clip:
            # Ensure the frame is a numpy array
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            video.write(frame)
    
    video.release()
    return output_path

def load_custom_model_with_checkpoint(model_type):
    load_checkpoint = True
    if model_type == 'Custom ResNet + DistilBERT Model':
        ckpt_url = 'https://drive.google.com/file/d/1YVByA2v4AnnZ67vrdY13X5tLl-0w-OCy/view?usp=sharing'  # Replace with your checkpoint URL
        ckpt_path = "final.pt"
        model = Model().to(device)
        print("device  :",device)

    else:
        ckpt_url = 'https://drive.google.com/file/d/1YVByA2v4AnnZ67vrdY13X5tLl-0w-OCy/view?usp=sharing'
        ckpt_path = 'vit.pt'
        model = ViT_Model().to(device)
        print("device  :",device)


    # # Download the checkpoint file if needed
    # if not os.path.exists(ckpt_path):
    #     gdown.download(ckpt_url, ckpt_path, quiet=False)

    # Load the model
   
    # Load the checkpoint if specified and available
    if load_checkpoint and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))

        print("Loaded checkpoint")

    return model

def run_clip_search(frames, search_queries):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    batch_size = 32
    num_frames = len(frames)
    logits = np.zeros((num_frames, len(search_queries)))  # Ensure logits is 2D

    # Precompute text features
    text_features = clip_processor(text=search_queries, return_tensors="pt", padding=True).to(device)

    # Process in batches
    for i in range(0, num_frames, batch_size):
        batch_frames = frames[i:i + batch_size]
        image_features = clip_processor(images=batch_frames, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**text_features, pixel_values=image_features['pixel_values'])
        batch_logits = outputs.logits_per_image.squeeze().detach().cpu().numpy()

        # Check the output shape and reshape if necessary
        if batch_logits.ndim == 1:
            batch_logits = batch_logits.reshape(-1, len(search_queries))

        logits[i:i + batch_size] = batch_logits

    best_frames_idx = np.argmax(logits, axis=0)
    return best_frames_idx

# Display and search functionality for custom model
def run_custom_model_search(frames, search_texts, model_type):
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    model = load_custom_model_with_checkpoint(model_type)  # Load the custom model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    texts = tokenizer(search_texts, return_tensors='pt', padding=True, truncation=True).to(device)

    # Prepare transformation for frames
    transform = Compose([
        Resize((224, 224)),  # Resize frames to 224x224
        ToTensor(),          # Convert image to tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    batch_size = 32
    num_frames = len(frames)
    logits = np.zeros((num_frames, len(search_texts)))

    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            batch_frames = [transform(frame).unsqueeze(0) for frame in frames[i:i + batch_size]]
            batch_frames = torch.cat(batch_frames).to(device)

            # Feed into the model
            outputs = model(batch_frames, texts)
            logits[i:i + batch_size] = outputs.detach().cpu()

    best_frames_idx = np.argmax(logits, axis=0)
    return best_frames_idx


def display_results(frames, best_frames_idx, search_queries):
    for idx, query in zip(best_frames_idx, search_queries):
        st.image(frames[idx], caption=f"Best frame for: {query}")

# Streamlit UI
st.title("Video Content Search")

# Model selection
model_option = st.radio(
    "Select the model you want to use:",
    ('Pretrained CLIP Model', 'Custom ResNet + DistilBERT Model', 'Custom ViT + DistilBERT Model')
)

url = st.text_input("Enter a YouTube URL (or leave blank to upload a video):")
uploaded_file = st.file_uploader("Or upload a video file", type=["mp4", "avi", "mov", "mkv"])

search_queries = st.text_input("Enter search queries separated by commas").split(",")

genre = ""
probability_threshold = 0.5

genre = st.text_input("Enter the genre for the clip: (optional)")
probability_threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.5)

if st.button("Process Video"):
    if url:
        video_path = download_youtube_video(url)
    elif uploaded_file:
        # Save the uploaded file to a temporary location first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())  # use getvalue() for BytesIO, read() for File
            video_path = tmpfile.name
    else:
        st.warning("Please upload a file or enter a URL.")
        st.stop()

    frames, fps = load_video(video_path)
    if not frames or fps is None:
        st.error("Error loading video or extracting FPS.")
    if frames:
        if model_option == 'Pretrained CLIP Model' and search_queries!=['']:
            best_frames_idx = run_clip_search(frames, search_queries)
            display_results(frames, best_frames_idx, search_queries)
        elif model_option == 'Custom ResNet + DistilBERT Model' or model_option == 'Custom ViT + DistilBERT Model':
            best_frames_idx = run_custom_model_search(frames, search_queries, model_option)
            display_results(frames, best_frames_idx, search_queries)
        if genre:
            if model_option == 'Pretrained CLIP Model':
                clip_starts = run_clip_search(frames, [genre])
            elif model_option == 'Custom ResNet + DistilBERT Model' or model_option == 'Custom ViT + DistilBERT Model':
                clip_starts = run_custom_model_search(frames, [genre],model_option)
            
            frame_indices = clip_starts
            if frame_indices:
                clip_output_path = f"output_{genre}_clip.mp4"
                result_path = frames_to_video(frames, clip_starts, fps, clip_output_path)
                converted_path = result_path.replace('.mp4', '_h264.mp4')
                print(converted_path,'converted path')
                convert_video_to_h264(result_path, converted_path)
                if os.path.exists(converted_path):
                    st.video(converted_path)
                else:
                    st.error("Converted video file is missing.")
            else:
                st.warning("No video was generated.")
        else:
            st.warning("No clips found for the specified genre.")
    else:
        st.warning("Please enter valid search queries.")