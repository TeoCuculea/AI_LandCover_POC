"""Ziua 1, exercițiul 2: ce se întâmplă dacă uiți `model.eval()`.

Scriptul e complet. Tu doar PREZICI, apoi rulezi și compari.

Modelul antrenat (split aleator, RGB) este evaluat pe 640 de imagini de
validare în patru feluri:
    A. model.eval(),  loturi de 32   (corect)
    B. model.train(), loturi de 32   (ai uitat eval)
    C. model.eval(),  loturi de 2
    D. model.train(), loturi de 2

Scrie-ți pe hârtie, ÎNAINTE de a rula:
  1. Care dintre A–D dau exact aceeași acuratețe? De ce?
  2. Care va fi cea mai slabă? De ce?
Indiciu: batch norm folosește media LOTULUI în modul train și `running_mean`
(media ținută minte din antrenare) în modul eval.

Rulare (din rădăcina proiectului):
    python -m exercises.ziua_01_ex2_train_vs_eval
"""

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import EuroSATDataset
from src.models.resnet import build_model
from src.run_paths import run_directory

RUN_ID = "disertatie_final_01"
IMAGES = 640


def accuracy(model, dataset, batch_size: int, device) -> float:
    """Share of correct predictions. No gradients are computed either way."""
    correct = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            outputs = model(images.to(device))
            correct += (outputs.argmax(dim=1).cpu() == labels).sum().item()
    return correct / len(dataset)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = run_directory(RUN_ID) / "models" / "random_rgb_best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Lipsește modelul antrenat: {checkpoint_path}")

    validation = EuroSATDataset(split="validation", representation="rgb",
                                split_strategy="random")
    subset = Subset(validation, range(IMAGES))

    cases = (
        ("A", "eval", 32),
        ("B", "train", 32),
        ("C", "eval", 2),
        ("D", "train", 2),
    )
    for name, mode, batch_size in cases:
        # The checkpoint is reloaded each time: in train mode batch norm also
        # updates running_mean, which would change the following cases.
        model = build_model(representation="rgb", pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train() if mode == "train" else model.eval()

        result = accuracy(model, subset, batch_size, device)
        print(f"{name}. model.{mode}(), loturi de {batch_size:2d}: acuratețe {result:.4f}")

    # După ce ai rulat:
    #  a) A și C au ieșit identice? Ce spune asta despre modul eval?
    #  b) De ce e D mai slab decât B?
    #  c) torch.no_grad() a fost activ în toate cele patru cazuri. Ce face el
    #     și ce NU face?


if __name__ == "__main__":
    main()
