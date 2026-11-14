# AI Land Cover POC

Clasificarea acoperirii terenului pe setul EuroSAT (imagini Sentinel-2 de 64×64 pixeli, 10 clase) cu ResNet18. Proiectul compară două lucruri:

- **ce vede modelul**: 3 benzi RGB sau toate cele 13 benzi spectrale;
- **cum se împart datele**: split aleator sau split spațial, în care imaginile care se suprapun pe teren rămân în același subset.

Documentația completă, cu rezultate și concluzii: [documentation/EDA_Disertatie_CUCULEA_TEODOR-IOAN.docx](documentation/EDA_Disertatie_CUCULEA_TEODOR-IOAN.docx).

## Cum curge aplicația

Fiecare pas citește ce a scris pasul dinainte. Toate comenzile se rulează din rădăcina proiectului.

| Pas | Comandă | Ce face | Ce scrie |
|---|---|---|---|
| 1 | `python -m src.data.prepare` | Găsește imaginile, ia eticheta din numele folderului, face splitul aleator stratificat 70/15/15 și caută suprapuneri între subseturi | `results/tables/dataset_splits.csv`, `spatial_leakage_pairs.csv` |
| 2 | `python -m src.geospatial.create_spatial_split` | Grupează imaginile care se suprapun pe teren și pune fiecare grup întreg într-un singur subset | `results/tables/spatial_dataset_splits.csv` |
| 3 | `python -m src.data.normalization` | Calculează media și abaterea standard pe fiecare bandă, **doar din imaginile de antrenare** | `results/tables/normalization_statistics.json` |
| 4 | `python -m src.train ...` | Antrenează ResNet18, păstrează cel mai bun model după validare, îl evaluează o singură dată pe test | `results/runs/<run-id>/` |
| 5 | `python -m src.analysis.compare_experiments --run-id <run-id>` și celelalte rapoarte | Tabele și grafice din rezultatele salvate, fără reantrenare | `results/runs/<run-id>/tables`, `figures` |

Pașii 1–3 sunt deja rulați, iar fișierele lor sunt în repository. **Nu îi rula din nou** dacă vrei să reproduci rezultatele din documentație: pasul 1 și pasul 2 rescriu splitul.

## Structura

```text
src/
  config.py                 Căi și constante (benzi, proporții de split, seed)
  run_paths.py              Unde se scrie o rulare: results/runs/<run-id>/
  train.py                  Intrare scurtă către src/training/runner.py
  data/
    prepare.py              Pasul 1: EuroSATDataManager, splitul aleator
    normalization.py        Pasul 3: statistici de normalizare
    dataset.py              EuroSATDataset: citește o imagine și o dă modelului
  geospatial/
    create_spatial_split.py Pasul 2: SpatialSplitManager, splitul spațial
    leakage.py              Suprapuneri între subseturi
    coordinates.py, spatial_analysis.py, visualization.py, run_analysis.py
                            Coordonate, vecini apropiați, hartă interactivă
    plot_split_figures.py   Figurile care explică cele două splituri
  models/
    resnet.py               ResNet18Classifier: ResNet18 adaptat la 3 sau 13 benzi
  training/
    runner.py               Pasul 4: bucla de antrenare, early stopping, checkpoint
    metrics.py              Evaluare, raport de clasificare, matrice de confuzie
    plots.py                Curbele unei singure rulări
  analysis/
    exploration.py, spectral_signatures.py   Analiza exploratorie
    compare_experiments.py, compare_class_performance.py, plot_run_figures.py
                            Pasul 5: rapoartele unei rulări
    plot_style.py           Aspectul comun al figurilor
  diagnostics/              Verificări rapide (model, dataset, statistici pe benzi)
documentation/
  build_documentation.py    Generează documentul Word din rezultatele salvate
results/
  tables/, figures/         Splituri, normalizare, figuri de explorare
  runs/<run-id>/            Câte un folder pentru fiecare rulare de experimente
archive/                    Versiuni vechi, păstrate doar local (nu sunt în Git)
```

## Clasele importante

- **`EuroSATDataManager`** ([src/data/prepare.py](src/data/prepare.py)): splitul aleator. `split_dataset` apelează `train_test_split` de două ori: întâi 70% antrenare / 30% rest, apoi restul în două jumătăți, validare și test. `stratify` păstrează aceeași proporție de clase în fiecare subset.
- **`SpatialSplitManager`** ([src/geospatial/create_spatial_split.py](src/geospatial/create_spatial_split.py)): splitul spațial. `identify_spatial_groups` leagă imaginile care se suprapun (structura `UnionFind`), iar `assign_groups_to_splits` pune fiecare grup, cu totul, în antrenare, validare sau test.
- **`EuroSATDataset`** ([src/data/dataset.py](src/data/dataset.py)): `__getitem__` citește un TIFF, alege benzile (3 sau 13), scade media și împarte la abaterea standard, apoi întoarce tensorul și eticheta.
- **`ResNet18Classifier`** ([src/models/resnet.py](src/models/resnet.py)): ResNet18 preantrenat pe ImageNet. `_adapt_first_convolution` înlocuiește primul strat ca să accepte 13 benzi; ultimul strat este înlocuit cu unul de 10 clase.
- **`train_model`** ([src/training/runner.py](src/training/runner.py)): o epocă de antrenare, o evaluare pe validare, salvarea modelului dacă macro-F1 a crescut, oprire după 6 epoci fără progres. La final, modelul salvat este evaluat pe test.

## Instalare și date

```powershell
python -m pip install -r requirements.txt
```

Imaginile nu sunt în repository (3 GB). Arhiva EuroSAT multispectrală se descarcă de pe [Zenodo](https://zenodo.org/records/7711810) și se dezarhivează astfel încât clasele să fie în `data/EuroSAT_MS/EuroSAT_MS/<Clasă>/*.tif`.

## Rularea experimentelor

O singură antrenare:

```powershell
python -m src.train --representation rgb --split random --epochs 30 --run-id proba_01
```

Toate cele patru configurații, una după alta, apoi rapoartele:

```powershell
.\run_baseline_experiments.cmd --what-if --run-id baseline_02
.\run_baseline_experiments.cmd --run-id baseline_02 --epochs 30 --seed 42
```

`--what-if` doar afișează comenzile. Un `run-id` existent nu se suprascrie niciodată; `--continue-run` sare peste experimentele deja terminate. Pe un RTX 5060 Ti o epocă durează aproximativ două minute, deci cele patru configurații iau între două și patru ore.

Rapoartele pentru o rulare existentă:

```powershell
python -m src.analysis.compare_experiments --run-id disertatie_final_01
python -m src.analysis.compare_class_performance --run-id disertatie_final_01
python -m src.analysis.plot_run_figures --run-id disertatie_final_01
```

Figurile de explorare și cele despre splituri:

```powershell
python -m src.analysis.exploration
python -m src.analysis.spectral_signatures
python -m src.geospatial.plot_split_figures
```

Documentul Word:

```powershell
python documentation/build_documentation.py
```

## De știut

- Rezultatele din documentație vin din `results/runs/disertatie_final_01/`. Fișierele de model (`*.pth`, 134 MB fiecare) rămân doar local.
- Fișierele TIFF țin benzile în ordinea `B01 … B08, B09, B10, B11, B12, B8A`, cu B8A la sfârșit. `BAND_NAMES` din `src/config.py` respectă această ordine. În `results/tables/normalization_statistics.json` și `.csv`, generate înainte de corecție, **numele** ultimelor cinci benzi sunt în ordinea veche; valorile sunt corecte, pentru că se citesc după poziție.
- Coloana `image_path` din fișierele de split conține căi absolute de pe calculatorul pe care au fost generate. Antrenarea nu o folosește (construiește calea din clasă și numele fișierului); pasul 2 da.
