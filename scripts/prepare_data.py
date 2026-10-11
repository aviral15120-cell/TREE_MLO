"""Download CIFAR-10 and Cats-vs-Dogs (GitHub mirrors) and convert to 32x32 uint8 .npz files.

CIFAR-10 : github.com/YoongiKim/CIFAR-10-images  (50k/10k, stored as JPEG -> mild JPEG artefacts)
Cats/Dogs: github.com/laxmimerit/dog-cat-full-dataset (Kaggle split, 20k/5k), centre-cropped and
           resized to 32x32 with Lanczos so both datasets share one pipeline.
"""
import io, os, zipfile, urllib.request
import numpy as np
from PIL import Image

os.makedirs("data", exist_ok=True)
SRC = {
    "cifar": "https://codeload.github.com/YoongiKim/CIFAR-10-images/zip/refs/heads/master",
    "cd": "https://codeload.github.com/laxmimerit/dog-cat-full-dataset/zip/refs/heads/master",
}
CLS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def fetch(url, path):
    if not os.path.exists(path):
        print("downloading", url); urllib.request.urlretrieve(url, path)
    return zipfile.ZipFile(path)


z = fetch(SRC["cifar"], "data/cifar.zip")
for split in ["train", "test"]:
    X, Y = [], []
    for k, c in enumerate(CLS):
        for n in sorted(n for n in z.namelist() if n.startswith(f"CIFAR-10-images-master/{split}/{c}/") and n.endswith(".jpg")):
            X.append(np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB"))); Y.append(k)
    np.savez(f"data/cifar10_{split}.npz", X=np.stack(X), Y=np.array(Y))

z = fetch(SRC["cd"], "data/cd.zip")
for split in ["train", "test"]:
    X, Y = [], []
    for k, c in enumerate(["cats", "dogs"]):
        pre = f"dog-cat-full-dataset-master/data/{split}/{c}/"
        for n in sorted(n for n in z.namelist() if n.startswith(pre) and n.lower().endswith((".jpg", ".jpeg", ".png"))):
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
            w, h = im.size; s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((32, 32), Image.LANCZOS)
            X.append(np.array(im)); Y.append(k)
    np.savez(f"data/catsdogs_{split}.npz", X=np.stack(X), Y=np.array(Y))
print("done")
