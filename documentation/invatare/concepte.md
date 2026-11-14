# Fișe de concepte

Fiecare fișă are trei părți: ce este, unde apare în codul nostru și o întrebare la care să răspunzi cu voce tare, fără să te uiți la răspuns. Fișele se adaugă pe măsură ce avansăm. Aceleași întrebări sunt în [flashcards.csv](flashcards.csv), pentru Anki.

## Date și împărțirea lor

### Antrenare, validare, test
- **Ce este:** trei subseturi cu roluri diferite. Pe antrenare se schimbă ponderile. Pe validare se iau deciziile (la ce epocă te oprești, ce setări alegi). Testul se folosește o singură dată, la final, și dă nota cinstită.
- **În cod:** `split_dataset` din [prepare.py](../../src/data/prepare.py) apelează `train_test_split` de două ori (70/30, apoi 15/15). Rezultatul e coloana `split` din `results/tables/dataset_splits.csv`.
- **Întrebare:** de ce nu e de ajuns să ai doar antrenare și test?

### Stratificare
- **Ce este:** împărțire care păstrează în fiecare subset aceeași proporție de clase ca în setul întreg.
- **În cod:** argumentul `stratify=` din `train_test_split`.
- **Întrebare:** ce s-ar putea întâmpla cu clasa Pasture (2.000 de imagini) fără stratificare?

### Scurgere de informație (data leakage)
- **Ce este:** orice cale prin care informație din validare sau test ajunge în antrenare. Nota iese mai bună decât ar fi pe date cu adevărat noi.
- **În cod:** două exemple pe care le-am evitat: statisticile de normalizare se calculează doar din antrenare ([normalization.py](../../src/data/normalization.py)); imaginile suprapuse pe teren ajung în același subset ([create_spatial_split.py](../../src/geospatial/create_spatial_split.py)).
- **Întrebare:** dă un exemplu de scurgere care nu implică etichetele.

### Normalizare (standardizare)
- **Ce este:** pentru fiecare bandă, `(x − medie) / abatere`. Aduce benzile la aceeași scară, ca niciuna să nu domine doar pentru că are numere mari.
- **În cod:** `__getitem__` din [dataset.py](../../src/data/dataset.py).
- **Întrebare:** din ce subset vin media și abaterea, și de ce?

### Split spațial și Union-Find
- **Ce este:** imaginile care se suprapun sunt legate într-un grup, iar grupul merge întreg într-un singur subset. Union-Find ține evidența grupurilor: `find(x)` spune din ce grup face parte x, `union(a, b)` unește două grupuri.
- **În cod:** clasele `UnionFind` și `SpatialSplitManager` din [create_spatial_split.py](../../src/geospatial/create_spatial_split.py).
- **Întrebare:** A se suprapune cu B, B cu C, dar A nu cu C. Câte grupuri sunt și de ce?

## Rețeaua

### Convoluție
- **Ce este:** un filtru mic (de exemplu 3×3) care alunecă peste imagine și calculează, în fiecare poziție, o sumă ponderată. Numărul de filtre este numărul de canale de la ieșire. Pasul (stride) 2 înjumătățește rezoluția.
- **În cod:** prima convoluție din ResNet18 are 64 de filtre 7×7 cu pasul 2; ponderile ei au forma `[64, canale_intrare, 7, 7]`.
- **Întrebare:** câți parametri are o convoluție de la 13 la 64 de canale cu filtre 7×7, fără bias?

### Bloc rezidual
- **Ce este:** două convoluții 3×3 (cu batch norm și ReLU), iar la rezultat se adună intrarea: `ieșire = F(x) + x`. Adunarea îi dă gradientului un drum direct spre straturile de la început.
- **În cod:** `layer1` … `layer4` din ResNet18, câte două blocuri fiecare.
- **Întrebare:** ce se schimbă la primul bloc din `layer2`, `layer3` și `layer4`?

### Batch normalization
- **Ce este:** normalizează activările fiecărui canal. La antrenare folosește media și varianța lotului curent și își actualizează mediile mobile `running_mean` și `running_var`. La evaluare folosește mediile mobile, fixe. Are doi parametri învățați pe canal (scară și deplasare).
- **În cod:** `bn1` și straturile `bn` din fiecare bloc; comutarea se face cu `model.train()` și `model.eval()`.
- **Întrebare:** de ce depinde rezultatul de mărimea lotului dacă uiți `model.eval()`?

### Transfer learning și adaptarea la 13 benzi
- **Ce este:** pornești de la ponderi învățate pe alt set mare (ImageNet) în loc de ponderi aleatoare. Pentru 13 benzi, prima convoluție e înlocuită: filtrele pentru roșu, verde, albastru sunt copiate, celelalte pornesc de la zero.
- **În cod:** `_adapt_first_convolution` din [resnet.py](../../src/models/resnet.py).
- **Întrebare:** ce calculează modelul cu 13 benzi la primul pas de antrenare, față de cel RGB?

## Antrenarea

### Un pas de antrenare
- **Ce este:** cinci acțiuni, în ordine: ștergi gradientele vechi, forward, loss, backward, pasul optimizatorului.
- **În cod:** `train_one_epoch` din [runner.py](../../src/training/runner.py); îl scrii tu în [exercițiul 1](../../exercises/ziua_01_ex1_un_singur_lot.py).
- **Întrebare:** ce se întâmplă dacă nu ștergi gradientele?

### `model.eval()` și `torch.no_grad()`
- **Ce este:** două lucruri independente. `model.eval()` schimbă comportamentul straturilor (batch norm, dropout). `torch.no_grad()` oprește calculul gradientelor, ca să economisești timp și memorie.
- **În cod:** `evaluate_model` din [metrics.py](../../src/training/metrics.py) le folosește pe amândouă.
- **Întrebare:** care dintre ele schimbă predicțiile și care nu?

### Weight decay (AdamW)
- **Ce este:** la fiecare pas, ponderile sunt trase puțin spre zero. Ponderile mici dau o funcție mai netedă, deci mai puțin overfitting.
- **În cod:** `weight_decay=1e-4` la optimizator, în `train_model`.
- **Întrebare:** de ce este o formă de regularizare?

### Rata de învățare și cosine annealing
- **Ce este:** rata de învățare e mărimea pasului. Cosine annealing o scade treptat, după forma unui cosinus, de la valoarea inițială spre aproape zero.
- **În cod:** `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`.
- **Întrebare:** de ce și-au atins rulările lungi maximul abia în epocile 14–19?

### Early stopping
- **Ce este:** oprești antrenarea când scorul pe validare nu mai crește de un număr de epoci (răbdarea).
- **În cod:** `patience=6`, `min_delta=0.001` în `train_model`.
- **Întrebare:** de ce s-a oprit rularea „aleator · 13 benzi" la epoca 8, deși epoca 6 era cea mai bună de până atunci?

### Overfitting și underfitting
- **Ce este:** overfitting înseamnă scor mare pe antrenare și mai mic pe validare (modelul memorează). Underfitting înseamnă scor slab pe amândouă.
- **În cod:** la noi, ~99,9% pe antrenare față de 97–98% pe validare.
- **Întrebare:** numește două lucruri care reduc overfitting-ul și pe care încă nu le folosim.

## Evaluarea

### Acuratețe și macro-F1
- **Ce este:** acuratețea e procentul de imagini clasificate corect. Macro-F1 e media scorurilor F1 ale claselor, în care fiecare clasă cântărește la fel.
- **În cod:** `evaluate_model`; checkpoint-ul se alege după macro-F1 pe validare.
- **Întrebare:** când diferă mult cele două?

### Eroare standard și seed-uri
- **Ce este:** o acuratețe măsurată pe n imagini are zgomotul `sqrt(p(1−p)/n)`. Separat, rezultatul depinde de seed. O comparație serioasă raportează media ± abaterea pe mai multe seed-uri.
- **În cod:** [exercițiul 4](../../exercises/ziua_01_ex4_de_mana.py).
- **Întrebare:** de ce nu putem spune că 98,32% e mai bun decât 97,90%?

### IoU (intersecție supra reuniune)
- **Ce este:** pentru două măști, pixelii comuni împărțiți la pixelii din cel puțin una. 1 înseamnă identice, 0 înseamnă fără nimic în comun. Va fi metrica principală la segmentare.
- **În cod:** suprapunerea imaginilor din `leakage.py` folosește aceeași idee pe dreptunghiuri; pe măști o scrii tu în exercițiul 4.
- **Întrebare:** de ce nu folosim acuratețea când doar 1–2% din pixeli sunt „pierdere de pădure"?

## Algoritmi clasici

### K-means
- **Ce este:** grupare fără etichete. Alegi k, numărul de grupuri. Algoritmul repetă doi pași: fiecare punct merge la cel mai apropiat centru; fiecare centru se mută în media punctelor lui. Se oprește când nimic nu se mai schimbă. Rezultatul depinde de centrele de pornire, de aceea se fixează un seed.
- **În cod:** `cluster_geographic_locations` din [spatial_analysis.py](../../src/geospatial/spatial_analysis.py) grupează pozițiile imaginilor în 5 regiuni. E doar explorare; nu intră în antrenare.
- **La ce ne-ar folosi mai departe:** împărțirea patch-urilor în blocuri geografice pentru splitul pe regiuni.
- **Întrebare:** de ce k-means nu are nevoie de etichete, iar ResNet are?

### K-nearest neighbors
- **Ce este:** pentru un punct, găsești cele mai apropiate k puncte dintr-o mulțime. Ca metodă de clasificare, punctul primește clasa majoritară a vecinilor. Nu are antrenare propriu-zisă: „modelul" sunt datele.
- **În cod:** îl folosim doar pentru distanțe, nu pentru clasificare. `NearestNeighbors` din [analyze_spatial_distances.py](../../src/geospatial/analyze_spatial_distances.py) găsește, pentru fiecare imagine de test, cea mai apropiată imagine de antrenare (figura cu distanțele din documentație).
- **Întrebare:** care e diferența dintre k din k-means și k din k-nearest neighbors?

## Teledetecție

### Benzi și indici
- **Ce este:** Sentinel-2 măsoară lumina în 13 benzi. Vegetația sănătoasă reflectă puțin roșu și mult infraroșu apropiat. NDVI = (B08 − B04) / (B08 + B04) e mare pentru vegetație. NBR = (B08 − B12) / (B08 + B12) scade brusc după o tăiere sau un incendiu.
- **În cod:** `save_ndvi_example` din [exploration.py](../../src/analysis/exploration.py). NBR vine la baseline-ul pentru pierderi de pădure.
- **Întrebare:** de ce împărțim la sumă, în loc să folosim doar diferența B08 − B04?

### L1C și L2A
- **Ce este:** L1C e reflectanța la partea de sus a atmosferei (EuroSAT). L2A are corecție atmosferică, deci reflectanța la sol, și vine cu o hartă a norilor și umbrelor (stratul SCL). Pentru comparații între ani se folosește L2A.
- **Întrebare:** de ce contează corecția atmosferică atunci când compari aceeași zonă în doi ani diferiți?

### Mască
- **Ce este:** o imagine de aceeași mărime cu poza, în care fiecare pixel are o etichetă (1 = pierdere de pădure, 0 = rest). Aria se obține numărând pixelii: la 10 m, un pixel are 100 m².
- **Întrebare:** câte hectare înseamnă 5.000 de pixeli marcați?
