"""Compare handcrafted SVM, MobileNetV2 embedding SVM, and AP's original head."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src import config
from src.classical_ml.features import extract_folder, list_images
from src.classical_ml.train import macro_f1, select_and_train


class EmbeddingImages(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE[::-1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@torch.inference_mode()
def embed(model, paths, device):
    embedding_batches, head_batches = [], []
    ap_to_project = np.array([config.CLASS_TO_IDX[name] for name in sorted(config.CLASSES)])
    for inputs in DataLoader(EmbeddingImages(paths), batch_size=16):
        maps = model.features(inputs.to(device))
        vectors = F.adaptive_avg_pool2d(maps, (1, 1)).flatten(1)
        embedding_batches.append(vectors.cpu().numpy())
        head_batches.append(model.classifier(vectors).argmax(1).cpu().numpy())
    return np.concatenate(embedding_batches), ap_to_project[np.concatenate(head_batches)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=config.MODELS_DIR / "mobilenetv2_cucumber.pth")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"AP's MobileNetV2 checkpoint is required: {args.checkpoint}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config.CLASSES))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    root = config.PROCESSED_DIR
    if os.name == "nt" and root.drive and not str(root).startswith("\\\\"):
        root = Path("\\\\?\\" + str(root))
    with (config.ROOT_DIR / "data" / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))

    data = {}
    for split in config.SPLITS:
        listed, _ = list_images(root / split)
        expected = [(row["class"], row["filename"]) for row in manifest if row["split"] == split]
        actual = [(path.parent.name, path.name) for path in listed]
        if not expected or len(expected) != len(actual) or sorted(expected) != sorted(actual):
            raise ValueError(f"Processed {split} files differ from the shared manifest")
        features, labels, paths = extract_folder(root / split, desc=f"processed/{split}")
        embeddings, head_predictions = embed(model, paths, device)
        data[split] = (features, embeddings, labels, head_predictions)
        print(f"{split}: {len(labels)} images, {embeddings.shape[1]} embedding values each")

    rows = []
    for name, column in (("Handcrafted + SVM", 0), ("MobileNetV2 + SVM", 1),
                         ("MobileNetV2 + AP head", 3)):
        train, val, test = (data[split] for split in ("train", "val", "test"))
        if column == 3:
            params = {}
            val_predictions, test_predictions = val[3], test[3]
        else:
            svm, params, _ = select_and_train("svm", train[column], train[2], val[column], val[2])
            val_predictions, test_predictions = svm.predict(val[column]), svm.predict(test[column])
        row = {"model": name, "svm_c": params.get("C", ""),
               "val_macro_f1": macro_f1(val[2], val_predictions),
               "test_accuracy": accuracy_score(test[2], test_predictions),
               "test_macro_f1": macro_f1(test[2], test_predictions)}
        recalls = recall_score(test[2], test_predictions, labels=range(len(config.CLASSES)),
                               average=None, zero_division=0)
        row.update({f"test_recall_{label}": value for label, value in zip(config.CLASSES, recalls)})
        rows.append(row)
        print(f"{name}: val macro-F1 {row['val_macro_f1']:.4f}, "
              f"test accuracy {row['test_accuracy']:.2%}, test macro-F1 {row['test_macro_f1']:.4f}")

    output = config.RESULTS_DIR / "metrics" / "mobilenet_svm_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
