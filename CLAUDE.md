# AI-LandCover_POC

Proiectul de disertație al lui Teodor (master ML). Două părți în același repository:

- **`src/`** — exercițiul de încălzire, terminat: clasificare EuroSAT cu ResNet18, RGB vs. 13 benzi, split aleator vs. spațial. Rezultatele sunt în `results/runs/disertatie_final_01/`, documentul în `documentation/EDA_Disertatie_CUCULEA_TEODOR-IOAN.docx`. Nu se regenerează fișierele de split și de normalizare din `results/tables/`.
- **`forest/`** — POC-ul în lucru, termen **15 decembrie 2026**: detecția pierderilor de pădure din Carpații românești pe Sentinel-2. Etichete din EFDA v3.0 (30 m), imagini din Earth Search (10 m), totul în EPSG:3035.

Planul pe zile, cu ce e musai (M1–M8) și ce e opțional (N1–N8): [documentation/PLAN_POC.md](documentation/PLAN_POC.md). Starea fiecărui punct: [documentation/TRASABILITATE.md](documentation/TRASABILITATE.md).

## Cum se lucrează cu Teodor

- Răspunde în română. Explică prin flow și prin legătura cu ce știe deja, nu prin cod restructurat.
- O zi de lucru: teorie pe exemple mici, apoi el scrie o bucată delimitată, apoi citește ce ai scris tu, apoi 5 întrebări. Fiecare zi începe cu 5 întrebări din zilele trecute.
- Bucățile marcate „scrii tu" în plan **nu se rezolvă în locul lui**. Dai indicii, îi revizuiești încercarea, spui ce parte e corectă și corectezi raționamentul, nu doar concluzia.
- Regula lui: nu rămâne în proiect cod pe care nu-l poate explica.
- Când ceva amenință termenul, se taie din lista N, nu din M1–M8 și nu din înțelegere. Spune devreme.
- Fișele de concepte și întrebările se țin la zi în `documentation/invatare/`.

## Comenzi

Toate din rădăcina proiectului.

```powershell
# EuroSAT
python -m src.train --representation rgb --split random --epochs 30 --run-id <id>
python -m src.analysis.plot_run_figures --run-id disertatie_final_01
python documentation/build_documentation.py

# Pierderi de pădure
python -m forest.select_aoi
python -m forest.data.sentinel2 --candidate 1 --years 2023 2024
python -m forest.plot_quicklook --candidate 1 --year 2024

# Exerciții
python -m exercises.ziua_01_ex4_de_mana
```

## Convenții

- Cod, nume și comentarii în engleză; texte din figuri, documentație și exerciții în română.
- Figurile folosesc `src/analysis/plot_style.py` (`apply_style`, `save_figure`) și se salvează pe disc, nu se afișează.
- Cifrele din documente se citesc din fișierele de rezultate, nu se tastează.
- `data/`, `*.pth` și `archive/` nu sunt în Git. Datele pentru `forest/` stau în `data/forest/`.
- Grila comună: EPSG:3035; etichete la 30 m, imagini la 10 m, aliniate astfel încât 3×3 pixeli de imagine acoperă exact un pixel de etichetă.
