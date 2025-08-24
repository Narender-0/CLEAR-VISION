# CLEAR-VISION
CLEAR-VISION is a deep learning project for image restoration, designed to remove corruptions from images using a GAN-based architecture. It includes corruption module, original and altered datasets, training pipelines, validation scripts, pre-trained checkpoints and a Streamlit demo for quick testing. This project was prepared under the Coding Club of IIT Guwahati as one of their "Even semsester Projects'25".

## Project Overview  

1. **Data Collection**  
   - Images were scraped from the web using **Selenium-based automation scripts**.  
   - Collected images were organized into:  
     - `Training_Data/clean_images` -> for training  
     - `Validation_Data/val_clean_images` -> for validation  

2. **Corruption Generation**  
   - A custom **Corruption Module** was developed to synthetically degrade clean images.  
   - These corrupted versions simulate real-world distortions (blur, noise, artifacts, etc.).  
   - Final datasets:  
     - `Training_Data/corrupted__images`  
     - `Validation_Data/val_corrupted_images`  

3. **Model Training**  
   - We trained a **Pix2Pix GAN** (conditional GAN) to learn the mapping:  

      $$\large\hspace{1in} \[
     \text{Corrupted Image} \ \rightarrow \ \text{Restored Image}
     \]$$
     
   - The model was trained with paired datasets (`clean` vs `corrupted`).  
   - Multiple checkpoints were saved in `Final_Checkpoints/` for reproducibility.  

4. **Evaluation**  
   - On validation images, restored outputs are saved in:  
     - `Validation_Data/val_generated_images`  
   - Performance measured using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**.  

5. **Deployment**  
   - A **Streamlit app** (`Streamlit.py`) allows users to upload a corrupted image and get its restored version instantly.  
   - Sample images are provided in `sample_images/` for quick testing.  




# Model Performance

# Access
THE LINK TO THE WEB INTERFACE IS 
https://clear-vision-hfvzqmpvxbpmjndreblrag.streamlit.app/
