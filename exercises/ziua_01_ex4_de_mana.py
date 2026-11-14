"""Ziua 1, exercițiul 4: trei formule scrise de tine.

Scrie corpul celor trei funcții. La rulare, scriptul le verifică pe exemple
cu răspuns cunoscut și îți spune care trec. Calculează întâi exemplele pe
hârtie: dacă știi ce trebuie să iasă, codul e ușor.

Rulare (din rădăcina proiectului):
    python -m exercises.ziua_01_ex4_de_mana
"""

import math

import numpy as np


def standard_error(accuracy: float, images: int) -> float:
    """Eroarea standard a unei acurateți măsurate pe `images` imagini.

    Formula: sqrt( p * (1 - p) / n ), cu p = acuratețea (între 0 și 1).
    Exemplu din documentație: p = 0.98, n = 4050  →  aproximativ 0.0022.
    """
    # TODO
    raise NotImplementedError


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersecție supra reuniune între două măști cu valori 0 și 1.

    intersecția = câți pixeli sunt 1 în AMBELE măști
    reuniunea   = câți pixeli sunt 1 în CEL PUȚIN una
    Dacă reuniunea e 0 (ambele măști goale), întoarce 1.0.

    Indiciu: cu numpy, (mask_a == 1) & (mask_b == 1) dă o matrice True/False,
    iar .sum() numără valorile True. Pentru „cel puțin una" se folosește |.
    """
    # TODO
    raise NotImplementedError


def conv_parameters(in_channels: int, out_channels: int, kernel: int, bias: bool) -> int:
    """Numărul de parametri ai unei convoluții cu filtre kernel × kernel.

    Fiecare filtru are in_channels * kernel * kernel ponderi.
    Sunt out_channels filtre. Dacă bias=True, fiecare filtru are în plus un bias.
    """
    # TODO
    raise NotImplementedError


# ----------------------------------------------------------------------
# Verificări: nu modifica nimic de aici în jos.
# ----------------------------------------------------------------------

def _checks():
    a = np.array([[1, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    b = np.array([[0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    empty = np.zeros((4, 4), dtype=int)
    return (
        ("eroarea standard pentru 98% pe 4050 de imagini",
         lambda: math.isclose(standard_error(0.98, 4050), 0.0022, abs_tol=0.0001)),
        ("eroarea standard scade când ai mai multe imagini",
         lambda: standard_error(0.98, 40500) < standard_error(0.98, 4050)),
        ("IoU: două pătrate 2×2 decalate cu o coloană (2 comuni din 6)",
         lambda: math.isclose(iou(a, b), 2 / 6)),
        ("IoU: o mască față de ea însăși",
         lambda: math.isclose(iou(a, a), 1.0)),
        ("IoU: două măști goale",
         lambda: math.isclose(iou(empty, empty), 1.0)),
        ("IoU: nicio suprapunere",
         lambda: math.isclose(iou(a, 1 - a), 0.0)),
        ("prima convoluție ResNet18 pe RGB: 64 de filtre 7×7, fără bias",
         lambda: conv_parameters(3, 64, 7, bias=False) == 9408),
        ("aceeași convoluție pe 13 benzi",
         lambda: conv_parameters(13, 64, 7, bias=False) == 40768),
        ("o convoluție 3×3 de la 64 la 128 de canale, cu bias",
         lambda: conv_parameters(64, 128, 3, bias=True) == 73856),
    )


def main() -> None:
    passed = 0
    checks = _checks()
    for description, check in checks:
        try:
            ok = bool(check())
            verdict = "trece" if ok else "NU trece"
        except NotImplementedError:
            ok, verdict = False, "nescris încă"
        except Exception as error:  # show the mistake instead of stopping
            ok, verdict = False, f"eroare: {type(error).__name__}: {error}"
        passed += ok
        print(f"[{verdict:^12s}] {description}")
    print(f"\n{passed} din {len(checks)} verificări trec.")

    # După ce trec toate:
    #  a) Diferența dintre cele două convoluții (40768 − 9408) este exact
    #     diferența de parametri dintre modelul cu 13 benzi și cel RGB din
    #     documentație. Verifică în document.
    #  b) Cât e eroarea standard dacă testul ar avea doar 400 de imagini?


if __name__ == "__main__":
    main()
