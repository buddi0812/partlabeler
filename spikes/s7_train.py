"""S7 step 2: fine-tune RF-DETR Small on the car crops (White, station 3).

No horizontal flip: the classes are left/right from the car's point of view, and a mirrored
image would teach left_drl as right_drl. Colour augmentation is strong because the targets are
a black and a dark green car at another station.

    python spikes/s7_train.py --epochs 30
"""
import argparse
import time

from rfdetr import RFDETRSmall

AUG = {
    "HueSaturationValue": {"hue_shift_limit": 180, "sat_shift_limit": 50, "val_shift_limit": 40, "p": 0.7},
    "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.6},
    "ToGray": {"p": 0.1},
    "GaussianBlur": {"blur_limit": 3, "p": 0.2},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.2},
    "Affine": {"scale": (0.85, 1.15), "translate_percent": (-0.05, 0.05), "p": 0.4},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/example/s7_ds")
    ap.add_argument("--out", default="data/example/s7_runs/baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()

    t = time.perf_counter()
    model = RFDETRSmall()
    model.train(dataset_dir=args.dataset, output_dir=args.out, epochs=args.epochs, resolution=args.resolution,
                batch_size=args.batch, grad_accum_steps=args.accum, lr=1e-4, aug_config=AUG,
                early_stopping=True, early_stopping_patience=8, checkpoint_interval=5,
                tensorboard=False, progress_bar="tqdm", num_workers=2)
    print(f"training finished in {(time.perf_counter() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
