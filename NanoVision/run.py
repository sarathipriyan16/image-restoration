import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 1. MODEL ARCHITECTURE
# ================================================================

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
                [[
                    [[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]]
                ]],
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


# ================================================================
# 2. LOAD MODEL
# ================================================================

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

    return model, a, b


# ================================================================
# 3. PROCESS ONE IMAGE
# ================================================================

def restore_image(
    input_path,
    output_path,
    model,
    a,
    b,
    device
):

    arr = np.load(
        input_path
    ).astype(np.float32)

    # ------------------------------------------------------------
    # Normalize input to [0,1]
    # ------------------------------------------------------------

    if arr.max() > 2.0:
        arr = arr / 255.0

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    arr = np.clip(
        arr,
        0.0,
        1.0
    )

    # ------------------------------------------------------------
    # Remove singleton channel if present
    # ------------------------------------------------------------

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"Expected grayscale input with shape "
            f"(H,W) or (H,W,1), got {arr.shape}"
        )

    # ------------------------------------------------------------
    # Model normalization
    # ------------------------------------------------------------

    normalized = (
        arr - a
    ) / (
        b - a + 1e-6
    )

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0).to(
        device
    )

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    with torch.inference_mode():

        output = model(tensor)

    # ------------------------------------------------------------
    # Convert back to [0,1]
    # ------------------------------------------------------------

    output = output.squeeze(
        0,
        1
    ).cpu().numpy()

    output = (
        output * (b - a)
        + a
    )

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    output = np.clip(
        output,
        0.0,
        1.0
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------
    # Save .npy with SAME filename
    # ------------------------------------------------------------

    np.save(
        output_path,
        output
    )


# ================================================================
# 4. MAIN
# ================================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------

    if not os.path.isdir(input_dir):

        print(
            f"ERROR: Input directory does not exist:\n"
            f"{input_dir}"
        )

        sys.exit(1)

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Model path relative to run.py
    # ------------------------------------------------------------

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        script_dir,
        "models",
        "trial_18_model.pth"
    )

    if not os.path.isfile(model_path):

        print(
            f"ERROR: Model weights not found:\n"
            f"{model_path}"
        )

        sys.exit(1)

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Trial 18 Image Restoration")
    print("=" * 60)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device          : {device}")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load model ONCE
    # ------------------------------------------------------------

    model, a, b = load_model(
        model_path,
        device
    )

    # ------------------------------------------------------------
    # Find all .npy files
    # ------------------------------------------------------------

    input_files = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "*.npy"
            )
        )
    )

    if not input_files:

        print(
            "ERROR: No .npy files found."
        )

        sys.exit(1)

    print(
        f"Found {len(input_files)} input files."
    )

    # ------------------------------------------------------------
    # Process all files
    # ------------------------------------------------------------

    start_time = time.time()

    for index, input_path in enumerate(
        input_files,
        start=1
    ):

        filename = os.path.basename(
            input_path
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        restore_image(
            input_path,
            output_path,
            model,
            a,
            b,
            device
        )

        print(
            f"[{index}/{len(input_files)}] "
            f"{filename} -> {filename}"
        )

    total_time = (
        time.time() - start_time
    )

    print()
    print("=" * 60)
    print("RESTORATION COMPLETE")
    print("=" * 60)
    print(
        f"Images processed : "
        f"{len(input_files)}"
    )
    print(
        f"Total time       : "
        f"{total_time:.2f} seconds"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()