%%writefile app.py
import streamlit as st
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import ImageFilter,ImageStat
import cv2
import numpy as np





class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(self._contract_block(in_channels, features, use_batchnorm=False), ResidualBlock(features))
        self.down2 = nn.Sequential(self._contract_block(features, features*2), ResidualBlock(features*2))
        self.down3 = nn.Sequential(self._contract_block(features*2, features*4), ResidualBlock(features*4))
        self.down4 = nn.Sequential(self._contract_block(features*4, features*8), ResidualBlock(features*8))
        # Decoder
        self.up1 = nn.Sequential(self._expand_block(features*8, features*4), ResidualBlock(features*4))
        self.up2 = nn.Sequential(self._expand_block(features*8, features*2), ResidualBlock(features*2))
        self.up3 = nn.Sequential(self._expand_block(features*4, features), ResidualBlock(features))
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _contract_block(self, in_c, out_c, kernel_size=4, stride=2, padding=1, use_batchnorm=True):
        layers = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _expand_block(self, in_c, out_c, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        out = self.final(torch.cat([u3, d1], dim=1))
        return out


# Same transforms as training

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
])

def rescale_to_01(x):
    return (x + 1) / 2

def preprocess_image(img_pil, device):
    x = transform(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]
    return x

def postprocess_tensor(fake):
    fake_rescal = rescale_to_01(fake)
    fake_rescal = torch.clamp(fake_rescal, 0, 1)
    return T.ToPILImage()(fake_rescal.squeeze(0).cpu())



# Loading generator weights only

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = UNetGenerator().to(device)
WEIGHTS_PATH = "/content/final_generator_best.pth"   # <- put your file here

state = torch.load(WEIGHTS_PATH, map_location=device)
generator.load_state_dict(state)   # ✅ actually load weights
generator.eval()


# Test-Time Augmentation (TTA)

def restore_with_tta(img_pil, device):
    x = preprocess_image(img_pil, device)

    preds = []
    with torch.no_grad():
        # original
        preds.append(generator(x))

        # h-flip
        preds.append(torch.flip(generator(torch.flip(x, [3])), [3]))

        # v-flip
        preds.append(torch.flip(generator(torch.flip(x, [2])), [2]))

        # h+v flip
        preds.append(torch.flip(generator(torch.flip(x, [2,3])), [2,3]))

    avg_pred = torch.mean(torch.stack(preds, dim=0), dim=0)  # average results
    return postprocess_tensor(avg_pred)


# Streamlit UI

st.set_page_config(page_title="Image Restoration GAN", layout="wide")
st.markdown(
    """
    <style>
    /* Entire app background */
    body, .block-container {
        background: linear-gradient(135deg, #141e30, #243b55);
        color: #f5f6fa;
    }

    /* Title */
    h1 {
        color: #f39c12 !important;
        text-align: center;
        font-size: 42px !important;
        margin-bottom: 10px;
        font-family: "Trebuchet MS", sans-serif;
    }

    /* Subheaders */
    h2, h3, .stMarkdown, .stSubheader {
        color: #ecf0f1 !important;
    }

    /* Upload box */
    .stFileUploader {
        border: 2px dashed #f39c12;
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.08);
    }

    /* Fix Browse button */
    .stFileUploader button {
        color: black !important;
        background-color: #f39c12 !important;
        border-radius: 8px;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background-color: #f39c12;
        color: black;
        border-radius: 12px;
        font-weight: bold;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e67e22;
        color: white;
    }

    /* Download button */
    .stDownloadButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        font-size: 15px;
        padding: 8px 20px;
    }
    .stDownloadButton>button:hover {
        background-color: #1e8449;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("✨ Image Restoration Using GAN ")
st.caption("Coding Club Project ")


# Upload or Choose Sample

SAMPLE_IMAGES = [
    "sample_images/sample-1.jpg",
    "sample_images/sample-2.jpg",
    "sample_images/sample-3.jpg",
    "sample_images/sample-4.jpg",
    "sample_images/sample-5.jpg",
    "sample_images/sample-6.jpg",
    "sample_images/sample-7.jpg",
    "sample_images/sample-8.jpg",
    "sample_images/sample-9.jpg",
    "sample_images/sample-10.jpg",
    "sample_images/sample-11.jpg",
    "sample_images/sample-12.jpg",
    "sample_images/sample-13.jpg",
    "sample_images/sample-14.jpg",
    "sample_images/sample-15.jpg"

]

sample_choice = st.selectbox(
    "",  # no label
    [" Try a sample"] + SAMPLE_IMAGES
)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

img = None
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
elif sample_choice != " Try a sample":   # ✅ only load if it's not placeholder
    try:
        img = Image.open(sample_choice).convert("RGB")
    except FileNotFoundError:
        st.error(f"❌ Could not find file: {sample_choice}. Make sure it exists in the 'samples/' folder.")


# Processing

if img is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Corrupted Image")
        st.image(img, width=300)

    if st.button("🔮 Restore Image"):

        out_img = restore_with_tta(img, device)   # restore only if blurry

        with c2:
            st.subheader("Restored Image")
            st.image(out_img, width=300)

        # download option
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Restored",
            data=buf.getvalue(),
            file_name="restored.png",
            mime="image/png"
        )


