# Ghid de lectură

Nu sunt rezumate. Pentru fiecare lucrare ai ce să citești, cât durează și la ce întrebări să răspunzi **în scris, cu vorbele tale**, în timp ce citești. Răspunsurile le discutăm după. Dacă nu poți răspunde la o întrebare, acela e locul de recitit.

Ordinea e cea în care ai nevoie de ele în plan.

## 1. ResNet — He et al., 2016
https://arxiv.org/abs/1512.03385 · 45 de minute · pentru serile 16–20 noiembrie

**Citește:** secțiunea 1 (introducerea), secțiunea 3 întreagă (3.1–3.4), figurile 1, 2 și 3, tabelul 1. Secțiunea 4 doar răsfoită.

1. Figura 1 arată o rețea cu 56 de straturi care are eroare mai mare decât una cu 20, **și pe antrenare**. De ce nu e asta overfitting? Cum numesc autorii problema?
2. Ce învață un bloc rezidual: ieșirea dorită H(x) sau altceva? Scrie relația dintre H(x), F(x) și x.
3. Ce fac autorii când x și F(x) nu au aceeași formă (alt număr de canale, altă rezoluție)?
4. În tabelul 1, găsește coloana pentru 18 straturi. Câte blocuri are fiecare etapă și câte canale? Compară cu ce ai obținut la [exercițiul 3](../../exercises/ziua_01_ex3_forme.py).
5. Unde stă batch norm față de convoluție și față de ReLU?

## 2. U-Net — Ronneberger et al., 2015
https://arxiv.org/abs/1505.04597 · 30 de minute · pentru serile 16–20 noiembrie

**Citește:** secțiunile 1, 2 și 3, cu figura 1 lângă tine tot timpul.

1. Pe figura 1, urmărește cu degetul drumul unei imagini. Unde scade rezoluția, unde crește, și ce se întâmplă cu numărul de canale?
2. Ce duc săgețile gri orizontale de la stânga la dreapta, și ce s-ar pierde fără ele?
3. Prin ce diferă legătura din U-Net (concatenare) de cea din ResNet (adunare)?
4. Ce formă are ieșirea rețelei față de intrare? Ce înseamnă fiecare valoare din ea?
5. De ce pun autorii atât accent pe augmentări? Câte imagini de antrenare aveau?

## 3. EFDA — Viana-Soto și Senf, 2025
https://essd.copernicus.org/articles/17/2373/2025/ · o oră · înainte de 21 noiembrie

De aici vin etichetele noastre, deci trebuie să știi ce sunt și cât de bune sunt.

**Citește:** secțiunile 1, 2.3, 2.4, 2.5, 2.6 și 3.2. Restul răsfoit.

1. Din ce satelit și la ce rezoluție sunt făcute hărțile? Ce înseamnă asta pentru marginile unei tăieri văzute la 10 m?
2. Cum decide metoda că un pixel a fost perturbat într-un anumit an (2.4)? Cu ce compară anul respectiv?
3. Ce este „masca de pădure" (2.3) și de ce contează pentru noi?
4. Cum sunt atribuite cauzele (2.6) și care sunt cele trei categorii? Atribuirea e pe pixel sau pe petic?
5. Din 3.2: ce erori de omisiune și de comisiune au pixelii perturbați? Spune în cuvinte ce înseamnă fiecare pentru un model antrenat pe aceste etichete.

Atenție: versiunea pe care o folosim (3.0, 2026) a schimbat metoda de detecție cu una bazată pe deep learning; articolul descrie versiunea anterioară. Atribuirea cauzelor și masca de pădure au rămas cele din articol.

## 4. Hansen et al., 2013
https://www.science.org/doi/10.1126/science.1244693 · 20 de minute · înainte de 21 noiembrie

**Citește:** textul principal (e scurt) și figurile.

1. Cum definesc autorii „pădurea" și „pierderea"? Include definiția plantațiile și tăierile urmate de replantare?
2. Ce nu poate spune harta despre o pierdere (cauza, legalitatea, dacă e permanentă)?
3. De ce avem nevoie de EFDA dacă există deja Hansen pentru tot globul?

## 5. Daudt et al., 2018
https://arxiv.org/abs/1810.08462 · 30 de minute · pentru serile 23–27 noiembrie

**Citește:** secțiunea cu arhitecturile propuse și figura cu cele trei rețele; tabelele de rezultate răsfoite.

1. Care sunt cele trei feluri de a da unei rețele două imagini ale aceleiași zone, luate la date diferite?
2. Ce înseamnă „siameză": ce împart cele două ramuri?
3. În planul nostru, intrarea este „anul t și t+1 concatenate". Căreia dintre cele trei arhitecturi îi corespunde?
4. Ce se schimbă în legăturile de tip skip la varianta cu diferență față de cea cu concatenare?

## 6. Isaienkov et al., 2021
https://ieeexplore.ieee.org/abstract/document/9241044 · 40 de minute · pentru serile 23–27 noiembrie

Cea mai apropiată lucrare de tema ta: tăieri în Ucraina, pe Sentinel-2, cu U-Net.

1. De unde au etichetele și câte poligoane au?
2. Ce benzi și ce indici dau modelului?
3. Cum împart datele în antrenare și test? Este împărțirea lor ferită de problema pe care ai găsit-o tu la EuroSAT?
4. Ce metrici raportează și ce valori obțin? Notează-le: sunt reperul cu care îți vei compara rezultatele.
5. Ce spun că le-a creat probleme (nori, anotimp, zăpadă)?

## 7. Roberts et al., 2017
https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881 · 30 de minute · înainte de 28 noiembrie

**Citește:** introducerea, partea despre structura spațială și caseta cu recomandări.

1. De ce dă o validare încrucișată aleatoare scoruri prea bune pe date cu structură spațială?
2. Ce este validarea pe blocuri și cum se alege mărimea blocului?
3. Leagă de proiectul EuroSAT: ce ar fi trebuit să fie un bloc acolo?

## 8. SSL4EO-S12 — Wang et al., 2023
https://arxiv.org/abs/2211.07044 · doar dacă ajungem la N4

1. Ce înseamnă preantrenare auto-supervizată și de ce nu are nevoie de etichete?
2. De ce ar ajuta la problema celor 10 benzi care pornesc de la zero?
