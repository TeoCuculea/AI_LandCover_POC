"""Ziua 1, exercițiul 3: urmărește forma tensorului prin ResNet18.

Un lot de 4 imagini de 64×64 intră în rețea. După fiecare etapă, tensorul are
forma [lot, canale, înălțime, lățime]. Completează în PREDICTIONS ce formă
crezi că are după fiecare etapă, apoi rulează: scriptul îți spune unde ai
nimerit.

Reguli de care ai nevoie:
  - o convoluție sau un pooling cu pasul (stride) 2 înjumătățește înălțimea și lățimea;
  - numărul de canale de la ieșire = numărul de filtre al stratului;
  - etapele layer2, layer3, layer4 înjumătățesc rezoluția și dublează canalele;
  - layer1 nu schimbă nici rezoluția, nici canalele.

Rulare (din rădăcina proiectului):
    python -m exercises.ziua_01_ex3_forme
"""

import torch

from src.models.resnet import build_model

# Înlocuiește fiecare None cu un tuplu, de exemplu (4, 64, 32, 32).
PREDICTIONS = {
    "intrare (13 benzi)": (4, 13, 64, 64),   # exemplu completat
    "conv1 (64 filtre 7×7, pas 2)": None,
    "maxpool (pas 2)": None,
    "layer1": None,
    "layer2": None,
    "layer3": None,
    "layer4": None,
    "avgpool (medie globală)": None,
    "fc (strat liniar)": None,               # aici forma are doar 2 numere
}


def trace_shapes(representation: str, channels: int) -> dict:
    """Run one batch through the network and record the shape after each stage."""
    model = build_model(representation=representation, pretrained=False).eval()
    net = model.backbone
    shapes = {}

    with torch.no_grad():
        x = torch.randn(4, channels, 64, 64)
        shapes[f"intrare ({channels} benzi)"] = tuple(x.shape)
        x = net.relu(net.bn1(net.conv1(x)))
        shapes["conv1 (64 filtre 7×7, pas 2)"] = tuple(x.shape)
        x = net.maxpool(x)
        shapes["maxpool (pas 2)"] = tuple(x.shape)
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(net, name)(x)
            shapes[name] = tuple(x.shape)
        x = net.avgpool(x)
        shapes["avgpool (medie globală)"] = tuple(x.shape)
        x = net.fc(torch.flatten(x, 1))
        shapes["fc (strat liniar)"] = tuple(x.shape)
    return shapes


def main() -> None:
    actual = trace_shapes("multispectral", 13)

    if any(value is None for value in PREDICTIONS.values()):
        print("Completează mai întâi toate valorile None din PREDICTIONS.\n")

    correct = 0
    for stage, shape in actual.items():
        predicted = PREDICTIONS.get(stage)
        if predicted is None:
            verdict = "necompletat"
        elif tuple(predicted) == shape:
            verdict = "corect"
            correct += 1
        else:
            verdict = f"greșit, tu ai zis {tuple(predicted)}"
        shown = shape if predicted is not None else "?"
        print(f"{stage:32s} {str(shown):20s} {verdict}")
    print(f"\n{correct} din {len(actual)} corecte.")

    # După ce le-ai nimerit pe toate:
    #  a) Rulează trace_shapes("rgb", 3). Ce formă diferă față de 13 benzi? Doar una.
    #  b) De câte ori s-a înjumătățit rezoluția în total? (64 → ?)
    #  c) Pentru U-Net, de ce contează că păstrăm formele intermediare?


if __name__ == "__main__":
    main()
