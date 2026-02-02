"""
Inference entrypoints for the web backend.

Pipelines
---------
BMI >= 30:
  CycleGAN (test.py) -> translated image -> VNet2D -> 256x256 mask
  -> circumference(px) -> abdominal circumference(cm)

BMI < 30:
  U-Net -> 256x256 mask -> circumference(px) -> abdominal circumference(cm)

Notes
-----
- Segmentation output is converted with sigmoid + threshold (0.5).
- Circumference is computed on the 256x256 binary mask.
- Mask is resized back to the original image size for visualisation.
"""

import os
import sys
import uuid
import subprocess
import tempfile
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

os.environ["WANDB_MODE"] = "offline"


############################################################
# CONFIGURATION
############################################################

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

# CycleGAN paths and experiment name
CYCLEGAN_DIR = os.path.join(MODEL_DIR, "cyclegan")
CYCLEGAN_EXPERIMENT_NAME = "obese2nonobese"

# CycleGAN checkpoints and runtime folders
CYCLEGAN_CHECKPOINTS_DIR = os.path.join(MODEL_DIR, "weights", "cyclegan")
CYCLEGAN_RESULTS_BASE = os.path.join(MODEL_DIR, "runtime", "cyclegan_results")
CYCLEGAN_SINGLE_INPUT_DIR = os.path.join(MODEL_DIR, "runtime", "cyclegan_input")

# Segmentation model weights
VNET2D_MODEL_PATH = os.path.join(MODEL_DIR, "weights", "vnet", "vnet_segmentation.pt")
UNET_MODEL_PATH = os.path.join(MODEL_DIR, "weights", "unet", "unet_segmentation.pt")

# CycleGAN runtime arguments
CYCLEGAN_NUM_TEST = 1
CYCLEGAN_DIRECTION = "AtoB"
CYCLEGAN_MODEL_SUFFIX = "_A"
CYCLEGAN_NO_FLIP = True
CYCLEGAN_NO_DROPOUT = True
CYCLEGAN_LOAD_SIZE = 256
CYCLEGAN_CROP_SIZE = 256
CYCLEGAN_NETG = "resnet_6blocks"

# Shared inference parameters
IMAGE_SIZE = 256
THRESHOLD = 0.5

# VNet2D hyperparameters (configure this with training settings)
BASE_CHANNELS = 16
VNET_DROPOUT = 0.0

# Disk output location for masks
PREDICTED_MASK_DIR = os.path.join("uploads", "segmented_masks")

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug logging
TEST_LOGS = False


############################################################
# HELPERS
############################################################

def calculate_ellipse_circumference_from_np(mask_np: np.ndarray) -> Optional[float]:
    """
    Circumference (px) from largest contour in a binary mask.
    Ellipse fit if possible, otherwise arcLength. Returns None if empty.
    """
    if mask_np.dtype != np.uint8:
        mask = (mask_np > 0).astype(np.uint8) * 255
    else:
        mask = mask_np.copy()

    h, w = mask.shape[:2]
    white_px = int((mask > 0).sum())
    tlog(f"Mask received: shape=({h},{w}), white_pixels={white_px}")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(contour))
    tlog(f"Largest contour: points={len(contour)}, area={area:.1f}")

    if len(contour) < 5:
        perim = float(cv2.arcLength(contour, True))
        tlog(f"Fallback arcLength circumference(px)={perim:.2f}")
        return perim

    ellipse = cv2.fitEllipse(contour)
    (_, _), axes, _ = ellipse
    a = max(axes) / 2.0
    b = min(axes) / 2.0
    circumference = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    tlog(f"Ellipse fit: a={a:.2f}, b={b:.2f}, circumference(px)={circumference:.2f}")
    return float(circumference)


def preprocess_image_gray_to_tensor(path: str, size: int) -> torch.Tensor:
    """
    Disk input -> [1,1,H,W] tensor (grayscale, resized, normalized).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img_r = img_r.astype(np.float32) / 255.0
    x = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0)
    return x


def preprocess_gray_np_to_tensor(img_gray: np.ndarray, size: int) -> torch.Tensor:
    """
    In-memory grayscale -> [1,1,H,W] tensor (resized, normalized).
    """
    if img_gray is None:
        raise ValueError("img_gray is None")

    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    img_r = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_LINEAR)
    img_r = img_r.astype(np.float32) / 255.0
    x = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0)
    return x


def _make_output_mask_path() -> Tuple[str, str]:
    """
    Generate output mask filename/path under uploads/segmented_masks.
    """
    os.makedirs(PREDICTED_MASK_DIR, exist_ok=True)
    suffix = uuid.uuid4().hex
    mask_filename = f"predicted_mask_{suffix}.png"
    mask_path = os.path.join(PREDICTED_MASK_DIR, mask_filename)
    return mask_filename, mask_path


def save_mask_np(mask_np: np.ndarray, out_path: str, target_size=None):
    """
    Save binary mask to disk (optional resize with NEAREST).
    """
    if mask_np.dtype != np.uint8:
        mask_np = (mask_np > 0).astype(np.uint8) * 255

    if target_size is not None:
        mask_img = Image.fromarray(mask_np).resize(target_size, Image.NEAREST)
    else:
        mask_img = Image.fromarray(mask_np)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_img.save(out_path)


def mask256_to_overlay_png_bytes(mask_256: np.ndarray, target_size_wh: Tuple[int, int]) -> bytes:
    """
    Resize 256x256 mask to (W,H) with NEAREST and return PNG bytes.
    """
    if mask_256.dtype != np.uint8:
        mask_256 = (mask_256 > 0).astype(np.uint8) * 255

    w, h = int(target_size_wh[0]), int(target_size_wh[1])
    mask_img = Image.fromarray(mask_256).resize((w, h), Image.NEAREST)
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()


def tlog(msg: str):
    """
    Debug logger (enabled by TEST_LOGS)
    """
    if TEST_LOGS:
        print(f"[DEBUG] {msg}")


def short_file(p: str) -> str:
    """
    Return basename for logging.
    """
    return os.path.basename(p)


############################################################
# VNET2D ARCHITECTURE (BMI >= 30)
############################################################

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, n_convs: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        for _ in range(n_convs):
            layers.append(ConvBlock(ch, ch))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        out = out + x
        return self.act(out)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )
        self.res = ResidualBlock(out_ch, n_convs=n_convs, dropout=dropout)

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )
        self.res = ResidualBlock(out_ch, n_convs=n_convs, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Match skip size (center-crop if needed)
        if x.shape[-2:] != skip.shape[-2:]:
            sh, sw = skip.shape[-2], skip.shape[-1]
            xh, xw = x.shape[-2], x.shape[-1]
            dh = (sh - xh) // 2
            dw = (sw - xw) // 2
            skip = skip[:, :, dh:dh + xh, dw:dw + xw]

        x = x + skip
        x = self.res(x)
        return x


class VNet2D(nn.Module):
    """
    VNet2D segmentation model (BMI >= 30 branch).
    """
    def __init__(self, in_channels: int = 1, base: int = 16, dropout: float = 0.0):
        super().__init__()

        self.in_conv = nn.Sequential(
            ConvBlock(in_channels, base),
            ConvBlock(base, base),
        )
        self.in_res = ResidualBlock(base, n_convs=1, dropout=dropout)

        self.down1 = Down(base, base * 2, n_convs=2, dropout=dropout)
        self.down2 = Down(base * 2, base * 4, n_convs=2, dropout=dropout)
        self.down3 = Down(base * 4, base * 8, n_convs=3, dropout=dropout)

        self.bottom_down = nn.Sequential(
            nn.Conv2d(base * 8, base * 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ELU(inplace=True),
        )
        self.bottom_res = ResidualBlock(base * 16, n_convs=3, dropout=dropout)

        self.up3 = Up(base * 16, base * 8, n_convs=3, dropout=dropout)
        self.up2 = Up(base * 8, base * 4, n_convs=2, dropout=dropout)
        self.up1 = Up(base * 4, base * 2, n_convs=2, dropout=dropout)
        self.up0 = Up(base * 2, base, n_convs=1, dropout=dropout)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x0 = self.in_res(x0)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xb = self.bottom_down(x3)
        xb = self.bottom_res(xb)

        y3 = self.up3(xb, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        return self.out_conv(y0)


def load_vnet2d_model(model_path: str, device: torch.device, base_channels: int = 16, dropout: float = 0.0):
    """
    Load model weights and return an eval() model. Returns None if not found.
    """
    model = VNet2D(in_channels=1, base=base_channels, dropout=dropout)
    if not model_path or not os.path.exists(model_path):
        print(f"V-Net2D weights not found at {model_path}.")
        return None
    try:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"Loaded V-Net2D model from {model_path}")
        return model
    except Exception as e:
        print("Failed to load V-Net2D model:", e)
        return None


_VNET_MODEL: Optional[VNet2D] = None


def get_vnet_model() -> VNet2D:
    """
    Lazy-load and cache the model instance.
    """
    global _VNET_MODEL
    if _VNET_MODEL is None:
        _VNET_MODEL = load_vnet2d_model(
            VNET2D_MODEL_PATH,
            DEVICE,
            base_channels=BASE_CHANNELS,
            dropout=VNET_DROPOUT,
        )
        if _VNET_MODEL is None:
            raise RuntimeError(
                f"Could not load VNet2D weights from {VNET2D_MODEL_PATH}. Check the path and file name."
            )
    return _VNET_MODEL


############################################################
# U-NET ARCHITECTURE (BMI < 30)
############################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net segmentation model (BMI < 30 branch).
    """
    def __init__(self, in_channels=1, num_classes=1, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64, kernel_size=kernel_size)
        self.down_convolution_2 = DownSample(64, 128, kernel_size=kernel_size)
        self.down_convolution_3 = DownSample(128, 256, kernel_size=kernel_size)
        self.down_convolution_4 = DownSample(256, 512, kernel_size=kernel_size)

        self.bottle_neck = DoubleConv(512, 1024, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.up_convolution_1 = UpSample(1024, 512, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.up_convolution_2 = UpSample(512, 256, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.up_convolution_3 = UpSample(256, 128, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.up_convolution_4 = UpSample(128, 64, kernel_size=kernel_size, dropout_rate=dropout_rate)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


def load_unet_model(model_path: str, device: torch.device, in_channels=1, num_classes=1, kernel_size=3):
    """
    Load model weights and return an eval() model. Returns None if not found.
    """
    model = UNet(in_channels=in_channels, num_classes=num_classes, kernel_size=kernel_size)
    if not model_path or not os.path.exists(model_path):
        print(f"U-Net weights not found at {model_path}.")
        return None
    try:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"Loaded U-Net model from {model_path}")
        return model
    except Exception as e:
        print("Failed to load U-Net model:", e)
        return None


_UNET_MODEL: Optional[UNet] = None


def get_unet_model() -> UNet:
    """
    Lazy-load and cache the model instance.
    """
    global _UNET_MODEL
    if _UNET_MODEL is None:
        _UNET_MODEL = load_unet_model(UNET_MODEL_PATH, DEVICE)
        if _UNET_MODEL is None:
            raise RuntimeError(
                f"Could not load U-Net weights from {UNET_MODEL_PATH}. Check the path and file name."
            )
    return _UNET_MODEL


############################################################
# CYCLEGAN EXECUTION (BMI >= 30)
############################################################

def _run_cyclegan_test_single(input_image_path: str) -> str:
    """
    Run CycleGAN test.py on a single image.

    Returns translated output image path.
    """
    test_py = os.path.join(CYCLEGAN_DIR, "test.py")
    if not os.path.exists(test_py):
        raise RuntimeError(f"CycleGAN test.py not found at {test_py}")

    # Reset CycleGAN input directory to contain only the current image
    os.makedirs(CYCLEGAN_SINGLE_INPUT_DIR, exist_ok=True)
    for f in os.listdir(CYCLEGAN_SINGLE_INPUT_DIR):
        full = os.path.join(CYCLEGAN_SINGLE_INPUT_DIR, f)
        if os.path.isfile(full):
            os.remove(full)

    ext = os.path.splitext(input_image_path)[1]
    input_copy_name = f"input_single{ext}"
    input_copy_path = os.path.join(CYCLEGAN_SINGLE_INPUT_DIR, input_copy_name)
    Image.open(input_image_path).save(input_copy_path)

    args = [
        sys.executable, test_py,
        "--dataroot", CYCLEGAN_SINGLE_INPUT_DIR,
        "--name", CYCLEGAN_EXPERIMENT_NAME,
        "--model", "test",
        "--checkpoints_dir", CYCLEGAN_CHECKPOINTS_DIR,
        "--results_dir", CYCLEGAN_RESULTS_BASE,
        "--num_test", str(CYCLEGAN_NUM_TEST),
        "--direction", CYCLEGAN_DIRECTION,
        "--model_suffix", CYCLEGAN_MODEL_SUFFIX,
        "--load_size", str(CYCLEGAN_LOAD_SIZE),
        "--crop_size", str(CYCLEGAN_CROP_SIZE),
        "--netG", CYCLEGAN_NETG,
    ]
    if CYCLEGAN_NO_FLIP:
        args.append("--no_flip")
    if CYCLEGAN_NO_DROPOUT:
        args.append("--no_dropout")

    try:
        env = os.environ.copy()
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        proc = subprocess.run(
            args,
            cwd=CYCLEGAN_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        print(proc.stdout)
    except subprocess.CalledProcessError as e:
        print("CycleGAN test.py failed with output:")
        print(e.stdout if hasattr(e, "stdout") else str(e))
        raise RuntimeError("CycleGAN test.py failed")
    except Exception as e:
        raise RuntimeError(f"Error running CycleGAN test.py: {e}")

    # CycleGAN output path candidates
    candidate1 = os.path.join(CYCLEGAN_RESULTS_BASE, CYCLEGAN_EXPERIMENT_NAME, "test_latest", "images")
    candidate2 = os.path.join(CYCLEGAN_RESULTS_BASE, CYCLEGAN_EXPERIMENT_NAME, "test", "images")
    candidate3 = os.path.join(CYCLEGAN_RESULTS_BASE, CYCLEGAN_EXPERIMENT_NAME, "images")

    images_dir = None
    for c in (candidate1, candidate2, candidate3):
        if os.path.isdir(c):
            images_dir = c
            break

    if images_dir is None:
        raise RuntimeError(
            "Could not find CycleGAN images folder after run. Checked:\n"
            f"{candidate1}\n{candidate2}\n{candidate3}"
        )

    # Select translated output image (prefer files containing 'fake')
    files = sorted(os.listdir(images_dir))
    fake_files = [f for f in files if "fake" in f.lower()]
    if not fake_files and files:
        fake_files = [files[0]]

    if not fake_files:
        raise RuntimeError("No CycleGAN output images found")

    translated_path = os.path.join(images_dir, fake_files[0])
    if not os.path.exists(translated_path):
        raise RuntimeError(f"CycleGAN translated image not found: {translated_path}")

    return translated_path


############################################################
# MODEL INFERENCE HELPERS
############################################################

def _predict_mask_256_with_vnet(image_path_for_vnet_input: str) -> np.ndarray:
    """
    Run VNet2D on a grayscale image path and return a 256x256 binary mask (0/255).
    """
    vnet_model = get_vnet_model()
    x = preprocess_image_gray_to_tensor(image_path_for_vnet_input, IMAGE_SIZE).to(DEVICE)

    with torch.no_grad():
        logits = vnet_model(x)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    mask_256 = (prob > THRESHOLD).astype(np.uint8) * 255

    tlog(f"VNet input path received: {short_file(image_path_for_vnet_input)}")
    white_px = int((mask_256 > 0).sum())
    tlog(f"VNet output mask generated: shape={mask_256.shape}, white_pixels={white_px}")

    return mask_256


def _predict_mask_256_with_unet_from_gray_np(img_gray: np.ndarray) -> np.ndarray:
    """
    Run U-Net on an in-memory grayscale image and return a 256x256 binary mask (0/255).
    """
    unet_model = get_unet_model()
    x = preprocess_gray_np_to_tensor(img_gray, IMAGE_SIZE).to(DEVICE)
    tlog(f"Preprocessed tensor: shape={tuple(x.shape)}")

    with torch.no_grad():
        logits = unet_model(x)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    mask_256 = (prob > THRESHOLD).astype(np.uint8) * 255

    white_px = int((mask_256 > 0).sum())
    tlog(f"U-Net mask generated: shape={mask_256.shape}, white_pixels={white_px}")

    return mask_256


def _predict_mask_256_with_unet(image_path_for_unet_input: str) -> np.ndarray:
    """
    Disk-based wrapper for U-Net (kept for compatibility with older callers).
    """
    img = cv2.imread(image_path_for_unet_input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path_for_unet_input}")
    return _predict_mask_256_with_unet_from_gray_np(img)


def _save_overlay_and_measure_disk(
    original_uploaded_image_path: str,
    mask_256: np.ndarray,
) -> Tuple[str, float]:
    """
    Compute circumference and save resized mask overlay to disk.
    """
    predicted_pixels = calculate_ellipse_circumference_from_np(mask_256)
    if predicted_pixels is None:
        raise ValueError("Could not detect abdomen contour from predicted mask.")

    mask_filename, mask_path = _make_output_mask_path()

    orig_img = Image.open(original_uploaded_image_path)
    orig_w, orig_h = orig_img.size

    save_mask_np(mask_256, mask_path, target_size=(orig_w, orig_h))
    return mask_filename, float(predicted_pixels)


############################################################
# DISK-BASED PIPELINES
############################################################

def run_pipeline_bmi_ge_30(
    image_path: str,
    bmi: float,
    scale_pixels_per_cm: float,
) -> Tuple[str, str, float]:
    """
    BMI >= 30: CycleGAN -> VNet2D -> disk overlay. Returns (ac_str, mask_filename, predicted_pixels)
    """
    if scale_pixels_per_cm <= 0:
        raise ValueError("Scale must be > 0 (pixels per cm).")

    translated_image_path = _run_cyclegan_test_single(image_path)
    tlog("BMI >= 30 pipeline selected")
    tlog(f"CycleGAN output produced: {short_file(translated_image_path)}")

    try:
        mask_256 = _predict_mask_256_with_vnet(translated_image_path)
        mask_filename, predicted_pixels = _save_overlay_and_measure_disk(image_path, mask_256)
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"VNet2D prediction failed for {translated_image_path}: {e}")

    ac_cm = float(predicted_pixels) / float(scale_pixels_per_cm)
    ac_str = f"{ac_cm:.2f}"

    print("=========== OUTPUT (BMI >= 30) ===========")
    print(f"BMI: {bmi}")
    print(f"Predicted pixels: {predicted_pixels}")
    print(f"Scale (pixels/cm): {scale_pixels_per_cm}")
    print(f"AC (cm): {ac_str}")
    print("=========================================")

    return ac_str, mask_filename, float(predicted_pixels)


def run_pipeline_bmi_lt_30(
    image_path: str,
    bmi: float,
    scale_pixels_per_cm: float,
) -> Tuple[str, str, float]:
    """
    BMI < 30: U-Net -> disk overlay. Returns (ac_str, mask_filename, predicted_pixels)
    """
    if scale_pixels_per_cm <= 0:
        raise ValueError("Scale must be > 0 (pixels per cm).")

    try:
        mask_256 = _predict_mask_256_with_unet(image_path)
        mask_filename, predicted_pixels = _save_overlay_and_measure_disk(image_path, mask_256)
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"U-Net prediction failed for {image_path}: {e}")

    ac_cm = float(predicted_pixels) / float(scale_pixels_per_cm)
    ac_str = f"{ac_cm:.2f}"

    print("=========== OUTPUT (BMI < 30) ===========")
    print(f"BMI: {bmi}")
    print(f"Predicted pixels: {predicted_pixels}")
    print(f"Scale (pixels/cm): {scale_pixels_per_cm}")
    print(f"AC (cm): {ac_str}")
    print("========================================")

    return ac_str, mask_filename, float(predicted_pixels)


############################################################
# IN-MEMORY PIPELINES (USED BY FASTAPI ROUTES)
############################################################

def run_pipeline_bmi_lt_30_inmemory(
    image_bgr: np.ndarray,
    bmi: float,
    scale_pixels_per_cm: float,
) -> Tuple[str, bytes, float]:
    """
    BMI < 30: U-Net in-memory. Returns (ac_str, overlay_png, predicted_pixels)
    """
    tlog("BMI < 30 pipeline selected (Direct U-Net)")
    tlog(f"Raw BGR input: shape={image_bgr.shape}")

    if scale_pixels_per_cm <= 0:
        raise ValueError("Scale must be > 0 (pixels per cm).")

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image array")

    orig_h, orig_w = image_bgr.shape[:2]
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    try:
        mask_256 = _predict_mask_256_with_unet_from_gray_np(img_gray)
        predicted_pixels = calculate_ellipse_circumference_from_np(mask_256)
        if predicted_pixels is None:
            raise ValueError("Could not detect abdomen contour from predicted mask.")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"U-Net prediction failed: {e}")

    ac_cm = float(predicted_pixels) / float(scale_pixels_per_cm)
    ac_str = f"{ac_cm:.2f}"

    tlog(f"Scale applied: pixels_per_cm={scale_pixels_per_cm:.2f} -> AC(cm)={ac_str}")

    print("=========== OUTPUT (BMI < 30) ===========")
    print(f"BMI: {bmi}")
    print(f"Predicted pixels: {predicted_pixels}")
    print(f"Scale (pixels/cm): {scale_pixels_per_cm}")
    print(f"AC (cm): {ac_str}")
    print("========================================")

    overlay_png = mask256_to_overlay_png_bytes(mask_256, (orig_w, orig_h))
    return ac_str, overlay_png, float(predicted_pixels)


def run_pipeline_bmi_ge_30_inmemory(
    image_bgr: np.ndarray,
    bmi: float,
    scale_pixels_per_cm: float,
) -> Tuple[str, bytes, float]:
    """
    BMI >= 30: CycleGAN + VNet2D in-memory. Returns (ac_str, overlay_png, predicted_pixels)
    """
    if scale_pixels_per_cm <= 0:
        raise ValueError("Scale must be > 0 (pixels per cm).")

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image array")

    orig_h, orig_w = image_bgr.shape[:2]

    with tempfile.TemporaryDirectory() as td:
        input_path = os.path.join(td, f"input_{uuid.uuid4().hex}.png")

        # Write temporary input image for CycleGAN
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(input_path)

        translated_image_path = _run_cyclegan_test_single(input_path)
        tlog("BMI >= 30 pipeline selected")
        tlog(f"CycleGAN output produced: {short_file(translated_image_path)}")

        try:
            mask_256 = _predict_mask_256_with_vnet(translated_image_path)
            predicted_pixels = calculate_ellipse_circumference_from_np(mask_256)
            if predicted_pixels is None:
                raise ValueError("Could not detect abdomen contour from predicted mask.")
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"VNet2D prediction failed for {translated_image_path}: {e}")

    ac_cm = float(predicted_pixels) / float(scale_pixels_per_cm)
    ac_str = f"{ac_cm:.2f}"

    tlog(f"Scale applied: pixels_per_cm={scale_pixels_per_cm:.2f} -> AC(cm)={ac_str}")

    print("=========== OUTPUT (BMI >= 30) ===========")
    print(f"BMI: {bmi}")
    print(f"Predicted pixels: {predicted_pixels}")
    print(f"Scale (pixels/cm): {scale_pixels_per_cm}")
    print(f"AC (cm): {ac_str}")
    print("=========================================")

    overlay_png = mask256_to_overlay_png_bytes(mask_256, (orig_w, orig_h))
    return ac_str, overlay_png, float(predicted_pixels)
