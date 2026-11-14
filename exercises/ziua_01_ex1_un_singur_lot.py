"""Ziua 1, exercițiul 1: supraînvață un singur lot.

Ideea: dacă bucla de antrenare e corectă, un model poate memora 32 de imagini
fără probleme, iar loss-ul ajunge aproape de zero. Dacă nu ajunge, bucla are
o greșeală. Este primul test pe care îl faci la orice proiect nou.

Ce ai de făcut: scrie corpul funcției `train_on_one_batch`. Restul e gata.

Rulare (din rădăcina proiectului):
    python -m exercises.ziua_01_ex1_un_singur_lot
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import EuroSATDataset
from src.models.resnet import build_model

STEPS = 60


def train_on_one_batch(model, images, labels, criterion, optimizer, steps):
    """Antrenează modelul de `steps` ori pe ACELAȘI lot și întoarce lista de loss-uri.

    La fiecare pas ai nevoie de cinci lucruri, în această ordine:
      1. șterge gradientele rămase de la pasul trecut   (optimizer....)
      2. forward: trece imaginile prin model             (outputs = ...)
      3. calculează loss-ul dintre outputs și labels     (loss = ...)
      4. backward: calculează gradientele                (loss....)
      5. actualizează ponderile                          (optimizer....)

    Adaugă `loss.item()` în listă după fiecare pas.
    Te poți uita la `train_one_epoch` din src/training/runner.py DUPĂ ce ai încercat singur.
    """
    losses = []
    # TODO: scrie aici bucla
    raise NotImplementedError("Scrie bucla de antrenare în train_on_one_batch.")
    return losses


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EuroSATDataset(split="train", representation="rgb", split_strategy="random")
    images, labels = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))
    images, labels = images.to(device), labels.to(device)
    print(f"Lot: {tuple(images.shape)}, etichete: {labels.tolist()}")

    model = build_model(representation="rgb", pretrained=True).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = train_on_one_batch(model, images, labels, criterion, optimizer, STEPS)

    for step in range(0, len(losses), 10):
        print(f"pas {step:3d}: loss = {losses[step]:.4f}")
    print(f"pas {len(losses) - 1:3d}: loss = {losses[-1]:.4f}")

    if len(losses) != STEPS:
        print(f"\nAi {len(losses)} valori, trebuiau {STEPS}: adaugi loss-ul la fiecare pas?")
    elif losses[-1] < 0.05:
        print("\nCorect: modelul a memorat lotul, deci bucla funcționează.")
    else:
        print("\nLoss-ul nu a coborât sub 0,05. Verifică ordinea celor cinci pași.")

    # Întrebări (răspunde înainte de a merge mai departe):
    #  a) Ce se întâmplă cu loss-ul dacă ștergi linia cu zero_grad? Încearcă.
    #  b) De ce NU este acest rezultat o dovadă că modelul e bun?


if __name__ == "__main__":
    main()
