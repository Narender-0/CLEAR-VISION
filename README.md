# CLEAR-VISION
**CLEAR-VISION** is a deep learning project for image restoration, designed to remove corruptions from images using a **GAN-based architecture**. It includes corruption module, original and altered datasets, training pipelines, validation scripts, pre-trained checkpoints and a Streamlit demo for quick testing. This project was developed as part of "*Even semsester Projects'25*", under the **Coding Club of IIT Guwahati**.

## Project Overview  

1. **Data Collection**
   - **54,380 images** were scraped from the web using **Selenium-based automation scripts**.  
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
     \text{Corrupted Image} \ \rightarrow \ \text{Generated Image}
     \]$$
     
   - The model was trained with paired datasets (`clean` vs `corrupted`).  
   - Multiple checkpoints were saved in `Final_Checkpoints/` for reproducibility.  

4. **Evaluation**  
   - On validation images, restored outputs are saved in:  
     - `Validation_Data/val_generated_images`  
   - Performance measured using **SSIM (Structural Similarity Index Measure)**, **MS-SSIM (Multi-Scale Structural Similarity Index Measure)**, **PSNR (Peak Signal-to-Noise Ratio)**, and **LPIPS (Learned Perceptual Image Patch Similarity)**.  

5. **Deployment**  
   - A **Streamlit app** (`Streamlit.py`) allows users to upload a corrupted image and get its restored version instantly.  
   - Sample images are provided in `sample_images/` for quick testing.  

## Model Performance

To evaluate the performance of our Pix2Pix GAN, we measured image restoration quality using standard metrics:
- *SSIM* - **0.9235**
- *MS-SSIM* - **0.9828**
- *PSNR* - **28.69**
- *LPIPS* - **0.0979**
- *Inference Latency* - **0.18 ms/image**

## Access
The link to the web interface is: 
https://clear-vision-hfvzqmpvxbpmjndreblrag.streamlit.app/
