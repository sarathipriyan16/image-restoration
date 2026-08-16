import os
import glob
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


# ======================================================================
# 1. MODEL ARCHITECTURE — TRIAL 18 (ConvNeXt-V2)
# ======================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(
            x.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous()


class GlobalResponseNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.beta = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x):

        gx = torch.norm(
            x,
            p=2,
            dim=(2, 3),
            keepdim=True
        )

        nx = gx / (
            gx.mean(dim=1, keepdim=True)
            + self.eps
        )

        return (
            self.gamma * (x * nx)
            + self.beta
            + x
        )


class ConvNeXtV2Block(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )

        self.norm = LayerNorm2d(dim)

        self.pwconv1 = nn.Conv2d(
            dim,
            4 * dim,
            1
        )

        self.act = nn.GELU()

        self.grn = GlobalResponseNorm(
            4 * dim
        )

        self.pwconv2 = nn.Conv2d(
            4 * dim,
            dim,
            1
        )

    def forward(self, x):

        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        return residual + x


class LaplacianEdgeExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "weight",
            torch.tensor(
                [
                    [
                        [
                            [0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]
                        ]
                    ]
                ],
                dtype=torch.float32
            )
        )

    def forward(self, x):

        return F.conv2d(
            x,
            self.weight.to(
                device=x.device,
                dtype=x.dtype
            ),
            padding=1
        )


class RepPhyDAS_ConvNeXt(nn.Module):

    def __init__(
        self,
        img_channel=1,
        width=32
    ):
        super().__init__()

        self.edge_extractor = (
            LaplacianEdgeExtractor()
        )

        self.intro = nn.Conv2d(
            2,
            width,
            3,
            padding=1
        )

        self.enc1 = nn.Sequential(
            ConvNeXtV2Block(width),
            ConvNeXtV2Block(width)
        )

        self.down = nn.Conv2d(
            width,
            width * 2,
            2,
            stride=2
        )

        self.bottleneck = nn.Sequential(
            ConvNeXtV2Block(width * 2),
            ConvNeXtV2Block(width * 2)
        )

        self.up = nn.ConvTranspose2d(
            width * 2,
            width,
            2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            ConvNeXtV2Block(width),
            ConvNeXtV2Block(width)
        )

        self.pixel_shuffle_up = nn.Sequential(
            nn.Conv2d(
                width,
                width * 4,
                3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                width,
                width,
                3,
                padding=1
            )
        )

        self.out_head = nn.Conv2d(
            width,
            img_channel,
            3,
            padding=1
        )

    def forward(self, y):

        edge_map = self.edge_extractor(y)

        x = self.intro(
            torch.cat(
                [y, edge_map],
                dim=1
            )
        )

        e1 = self.enc1(x)

        b = self.bottleneck(
            self.down(e1)
        )

        d1 = self.dec1(
            self.up(b) + e1
        )

        out = self.pixel_shuffle_up(d1)

        return torch.clamp(
            self.out_head(out),
            0.0,
            1.0
        )


# ======================================================================
# 2. LOAD MODEL
# ======================================================================

def load_model(model_path, device):

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    a = checkpoint["a"]
    b = checkpoint["b"]

    model = RepPhyDAS_ConvNeXt(
        width=32
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("==============================================")
    print("Trial 18 ConvNeXt-V2 model loaded")
    print(f"Normalization parameters: a={a:.6f}, b={b:.6f}")
    print(f"Device: {device}")
    print("==============================================")

    return model, a, b


# ======================================================================
# 3. SINGLE IMAGE INFERENCE
# ======================================================================

def infer_image(
    npy_path,
    model,
    a,
    b,
    device
):

    arr = np.load(
        npy_path
    ).astype(np.float32)

    # Convert 8-bit input to [0, 1] if required
    if arr.max() > 2.0:
        arr = arr / 255.0

    # Same normalization used during training
    in_norm = (
        arr - a
    ) / (
        b - a + 1e-6
    )

    in_tensor = torch.from_numpy(
        in_norm
    ).unsqueeze(0).unsqueeze(0).to(device)

    # CUDA operations are asynchronous.
    # Synchronization gives a more reliable inference measurement.
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        out_tensor = model(
            in_tensor
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inf_time = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    # Convert model output back to original intensity range
    out_arr = (
        out_tensor
        .squeeze()
        .cpu()
        .numpy()
    )

    out_denorm = (
        out_arr * (b - a)
        + a
    )

    out_img = np.clip(
        out_denorm * 255.0,
        0,
        255
    ).astype(np.uint8)

    return out_img, inf_time


# ======================================================================
# 4. MAIN EVALUATION FUNCTION
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Trial 18 ConvNeXt-V2 "
            "image restoration evaluator"
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing "
            "input .npy images"
        )
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Directory where restored "
            "images will be saved"
        )
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to trained model weights. "
            "Defaults to trial_18_model.pth "
            "in the same directory as this script."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Locate model
    # --------------------------------------------------------------

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    if args.model is None:

        model_path = os.path.join(
            script_dir,
            "trial_18_model.pth"
        )

    else:

        model_path = args.model

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            f"Model weights not found: "
            f"{model_path}"
        )

    # --------------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------------

    if not os.path.isdir(args.input_dir):

        raise NotADirectoryError(
            f"Input directory not found: "
            f"{args.input_dir}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Find input images
    # --------------------------------------------------------------

    input_files = sorted(
        glob.glob(
            os.path.join(
                args.input_dir,
                "*.npy"
            )
        )
    )

    input_files = [
        path
        for path in input_files
        if "__MACOSX" not in path
    ]

    if len(input_files) == 0:

        raise RuntimeError(
            "No .npy files were found in "
            f"{args.input_dir}"
        )

    print(
        f"\nFound {len(input_files)} input images."
    )

    # --------------------------------------------------------------
    # Select device
    # --------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    model, a, b = load_model(
        model_path,
        device
    )

    # --------------------------------------------------------------
    # Process images
    # --------------------------------------------------------------

    latency_list = []

    print("\nStarting inference...\n")

    for index, input_path in enumerate(
        input_files,
        start=1
    ):

        output_img, latency = infer_image(
            input_path,
            model,
            a,
            b,
            device
        )

        latency_list.append(
            latency
        )

        input_filename = os.path.basename(
            input_path
        )

        output_filename = (
            os.path.splitext(
                input_filename
            )[0]
            + "_restored.png"
        )

        output_path = os.path.join(
            args.output_dir,
            output_filename
        )

        # Save grayscale PNG
        Image.fromarray(output_img, mode="L").save(output_path)

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_filename} -> "
            f"{output_filename} | "
            f"{latency:.2f} ms"
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    average_latency = np.mean(
        latency_list
    )

    total_time = np.sum(
        latency_list
    )

    print("\n==============================================")
    print("TRIAL 18 EVALUATION COMPLETE")
    print("==============================================")
    print(
        f"Images processed : {len(input_files)}"
    )
    print(
        f"Average inference latency : "
        f"{average_latency:.2f} ms/image"
    )
    print(
        f"Total inference time : "
        f"{total_time:.2f} ms"
    )
    print(
        f"Output directory : "
        f"{os.path.abspath(args.output_dir)}"
    )
    print("==============================================")


if __name__ == "__main__":
    main()