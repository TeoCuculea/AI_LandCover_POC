# AI-LandCover_POC

Exercițiul de încălzire pentru disertația lui Teodor (master ML), **terminat**: clasificare EuroSAT cu ResNet18, RGB vs. 13 benzi, split aleator vs. spațial. Rezultatele sunt în `results/runs/disertatie_final_01/`, documentul în `documentation/EDA_Disertatie_CUCULEA_TEODOR-IOAN.docx`.

Disertația propriu-zisă (pierderi de pădure pe Sentinel-2, POC pe 15 decembrie 2026) este în alt proiect: `E:\Forest_Lost_Romania`, repository `TeoCuculea/forest-loss-romania`. Planul și starea lui sunt acolo.

Aici rămân materialele de învățare bazate pe acest cod: `documentation/invatare/` (fișe de concepte, flashcards, ghid de lectură) și `exercises/`.

## Cum se lucrează cu Teodor

- Răspunde în română. Explică prin flow și prin legătura cu ce știe deja, nu prin cod restructurat.
- Bucățile de exercițiu marcate „scrii tu" **nu se rezolvă în locul lui**. Dai indicii, îi revizuiești încercarea, spui ce parte e corectă și corectezi raționamentul, nu doar concluzia.
- Regula lui: nu rămâne în proiect cod pe care nu-l poate explica.

## Comenzi

Toate din rădăcina proiectului.

```powershell
python -m src.train --representation rgb --split random --epochs 30 --run-id <id>
python -m src.analysis.plot_run_figures --run-id disertatie_final_01
python documentation/build_documentation.py
python -m exercises.ziua_01_ex4_de_mana
```

## Convenții

- Nu se regenerează fișierele de split și de normalizare din `results/tables/`: de ele depinde reproducerea rezultatelor.
- Cod, nume și comentarii în engleză; texte din figuri, documentație și exerciții în română.
- Figurile folosesc `src/analysis/plot_style.py` și se salvează pe disc.
- Cifrele din document se citesc din fișierele de rezultate, nu se tastează.
- `/data/`, `*.pth` și `archive/` nu sunt în Git.
