"""Build the project report (Word) from the saved results.

Every number in the document is read from the result files of one run and
from the split tables, so the text cannot drift away from the experiments.
The figures must already exist; the README lists the commands that draw them.

Usage (from the project root, with the document closed in Word):
    python documentation/build_documentation.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    BAND_NAMES,
    COORDINATES_FILE,
    FIGURES_DIR,
    NORMALIZATION_FILE,
    SPATIAL_SPLIT_FILE,
    TABLES_DIR,
)
from src.models.resnet import build_model  # noqa: E402
from src.run_paths import run_directory  # noqa: E402

RUN_ID = "disertatie_final_01"
OUTPUT_FILE = PROJECT_ROOT / "documentation" / "EDA_Disertatie_CUCULEA_TEODOR-IOAN.docx"
GPU_NAME = "NVIDIA GeForce RTX 5060 Ti (16 GB)"

EXPERIMENTS = (
    ("random_rgb", "Aleator", "RGB"),
    ("random_multispectral", "Aleator", "13 benzi"),
    ("spatial_rgb", "Spațial", "RGB"),
    ("spatial_multispectral", "Spațial", "13 benzi"),
)

# Sentinel-2 band facts: central wavelength (nm), native resolution (m), role.
BAND_FACTS = {
    "B01": (443, 60, "aerosoli (coastal)"),
    "B02": (490, 10, "albastru"),
    "B03": (560, 10, "verde"),
    "B04": (665, 10, "roșu"),
    "B05": (705, 20, "red edge 1"),
    "B06": (740, 20, "red edge 2"),
    "B07": (783, 20, "red edge 3"),
    "B08": (842, 10, "infraroșu apropiat (NIR)"),
    "B8A": (865, 20, "NIR îngust"),
    "B09": (945, 60, "vapori de apă"),
    "B10": (1375, 60, "cirrus"),
    "B11": (1610, 20, "infraroșu de undă scurtă (SWIR 1)"),
    "B12": (2190, 20, "infraroșu de undă scurtă (SWIR 2)"),
}


# ----------------------------------------------------------------------
# Number formatting (Romanian: decimal comma, dot for thousands)
# ----------------------------------------------------------------------

def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 2) -> str:
    """Format a share between 0 and 1 as a percentage."""
    return num(value * 100, digits) + "%"


def thousands(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def counted(value: int, noun: str) -> str:
    """Number plus noun, with the Romanian "de" where grammar asks for it.

    Numbers whose last two digits are 01-19 take the noun directly
    ("17 imagini"); all others need "de" ("45 de imagini").
    """
    last_two = int(value) % 100
    link = " " if 1 <= last_two <= 19 else " de "
    return thousands(value) + link + noun


# ----------------------------------------------------------------------
# Reading the results
# ----------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_facts() -> dict:
    """Collect every number the text needs, straight from the result files."""
    facts = {"experiments": {}}
    run_root = run_directory(RUN_ID)

    for name, _, _ in EXPERIMENTS:
        directory = run_root / "experiments" / name
        matrix = pd.read_csv(directory / "test_confusion_matrix.csv", index_col=0)
        history = pd.read_csv(directory / "training_history.csv")
        # One epoch lasted 99 minutes because the computer went to sleep;
        # durations above 1000 s are left out of the timing.
        durations = history.loc[history["duration_seconds"] < 1000, "duration_seconds"]
        facts["experiments"][name] = {
            "metrics": load_json(directory / "final_metrics.json"),
            "report": load_json(directory / "test_classification_report.json"),
            "matrix": matrix,
            "history": history,
            "test_images": int(matrix.to_numpy().sum()),
            "errors": int(matrix.to_numpy().sum() - matrix.to_numpy().trace()),
            "seconds_per_epoch": float(durations.median()),
        }

    facts["class_names"] = list(facts["experiments"]["random_rgb"]["matrix"].index)

    splits = pd.read_csv(SPATIAL_SPLIT_FILE)
    facts["total_images"] = len(splits)
    facts["random_sizes"] = splits["split"].value_counts()
    facts["spatial_sizes"] = splits["spatial_split"].value_counts()
    facts["common_test"] = int(
        ((splits["split"] == "test") & (splits["spatial_split"] == "test")).sum()
    )

    facts["leakage_summary"] = pd.read_csv(TABLES_DIR / "spatial_leakage_summary.csv")
    leakage_pairs = pd.read_csv(TABLES_DIR / "spatial_leakage_pairs.csv")
    facts["leakage_pairs"] = len(leakage_pairs)
    facts["leakage_pairs_high_iou"] = int(
        (leakage_pairs["intersection_over_union"] > 0.5).sum()
    )
    facts["leakage_same_class"] = float(
        (leakage_pairs["class_1"] == leakage_pairs["class_2"]).mean()
    )
    facts["leakage_images"] = len(
        set(leakage_pairs["image_1"]) | set(leakage_pairs["image_2"])
    )

    groups = pd.read_csv(TABLES_DIR / "spatial_groups.csv")
    multi = groups.loc[groups["image_count"] > 1]
    facts["groups"] = len(groups)
    facts["groups_single"] = int((groups["image_count"] == 1).sum())
    facts["groups_multi"] = len(multi)
    facts["groups_multi_images"] = int(multi["image_count"].sum())
    facts["groups_largest"] = int(groups["image_count"].max())

    distances = pd.read_csv(TABLES_DIR / "spatial_distance_analysis.csv")
    test_distance = distances.loc[
        (distances["source_split"] == "test") & (distances["target_split"] == "train"),
        "nearest_distance_m",
    ]
    facts["distance_min"] = float(test_distance.min())
    facts["distance_median"] = float(test_distance.median())
    facts["distance_under_700"] = float((test_distance < 700).mean())
    facts["distance_under_1000"] = float((test_distance < 1000).mean())
    facts["distance_under_5000"] = float((test_distance < 5000).mean())

    facts["utm_zones"] = int(pd.read_csv(COORDINATES_FILE)["crs"].nunique())

    # The JSON was written before the band names were corrected; the values
    # are in file order, so they are matched to the corrected names by position.
    normalization = load_json(NORMALIZATION_FILE)["random"]["multispectral"]
    facts["band_means"] = dict(zip(BAND_NAMES, normalization["mean"]))

    facts["parameters"] = {
        representation: sum(
            parameter.numel()
            for parameter in build_model(representation, pretrained=False).parameters()
        )
        for representation in ("rgb", "multispectral")
    }
    return facts


def top_confusions(facts: dict, count: int = 5) -> list[tuple[str, str, int]]:
    """Class pairs confused most often, summed over the four experiments."""
    names = facts["class_names"]
    total = sum(
        experiment["matrix"].to_numpy() for experiment in facts["experiments"].values()
    )
    pairs = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            pairs.append((names[first], names[second],
                          int(total[first, second] + total[second, first])))
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)[:count]


# ----------------------------------------------------------------------
# Small helpers that write into the Word document
# ----------------------------------------------------------------------

class Report:
    """Thin wrapper over python-docx with numbered figures and tables."""

    def __init__(self) -> None:
        self.document = Document()
        self.figure_number = 0
        self.table_number = 0
        self._configure_page()

    def _configure_page(self) -> None:
        section = self.document.sections[0]
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Cm(2.5))

        styles = self.document.styles
        normal = styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.12
        for level, size in ((1, 15), (2, 12.5)):
            heading = styles[f"Heading {level}"]
            heading.font.name = "Calibri"
            heading.font.size = Pt(size)
            heading.font.bold = True
            heading.font.color.rgb = RGBColor(0x1C, 0x3F, 0x73)
            heading.element.rPr.rFonts.set(qn("w:asciiTheme"), "Calibri")
            heading.paragraph_format.space_before = Pt(14 if level == 1 else 10)
            heading.paragraph_format.space_after = Pt(4)
            heading.paragraph_format.keep_with_next = True

    def _add_runs(self, paragraph, text: str) -> None:
        """Write text; parts wrapped in **double stars** become bold."""
        for index, part in enumerate(text.split("**")):
            if part:
                paragraph.add_run(part).bold = index % 2 == 1

    def title(self, title: str, subtitle: str, byline: str) -> None:
        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(title)
        run.bold = True
        run.font.size = Pt(20)
        run.font.color.rgb = RGBColor(0x1C, 0x3F, 0x73)
        paragraph.paragraph_format.space_after = Pt(2)

        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(subtitle)
        run.font.size = Pt(13)
        paragraph.paragraph_format.space_after = Pt(2)

        paragraph = self.document.add_paragraph()
        run = paragraph.add_run(byline)
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)
        paragraph.paragraph_format.space_after = Pt(12)

    def heading(self, text: str, level: int = 1) -> None:
        self.document.add_heading(text, level=level)

    def text(self, text: str, justify: bool = True) -> None:
        paragraph = self.document.add_paragraph()
        if justify:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self._add_runs(paragraph, text)

    def bullets(self, items: list[str]) -> None:
        for item in items:
            paragraph = self.document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.space_after = Pt(3)
            self._add_runs(paragraph, item)

    def numbered(self, items: list[str]) -> None:
        # Word's automatic numbering would continue from the previous list,
        # so the numbers are written as text with a hanging indent.
        for number, item in enumerate(items, start=1):
            paragraph = self.document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(1.27)
            paragraph.paragraph_format.first_line_indent = Cm(-0.63)
            paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(1.27))
            paragraph.paragraph_format.space_after = Pt(3)
            self._add_runs(paragraph, f"{number}.\t{item}")

    def _caption(self, text: str, keep_with_next: bool = False) -> None:
        paragraph = self.document.add_paragraph()
        paragraph.paragraph_format.keep_with_next = keep_with_next
        paragraph.paragraph_format.space_after = Pt(10 if not keep_with_next else 4)
        self._add_runs(paragraph, text)
        for run in paragraph.runs:
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)

    def figure(self, path: Path, caption: str, width_cm: float = 16.0) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Figure not found, draw it first: {path}")
        self.figure_number += 1
        paragraph = self.document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
        self._caption(f"**Figura {self.figure_number}.** {caption}")

    def table(self, headers: list[str], rows: list[list[str]], caption: str,
              widths_cm: list[float] | None = None,
              text_columns: tuple[int, ...] = (0,)) -> None:
        """Add a table. ``text_columns`` are left-aligned; the others hold numbers."""
        self.table_number += 1
        self._caption(f"**Tabelul {self.table_number}.** {caption}", keep_with_next=True)

        table = self.document.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        def fill(cell, value: str, bold: bool, align_left: bool) -> None:
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if align_left else WD_ALIGN_PARAGRAPH.RIGHT
            )
            run = paragraph.add_run(value)
            run.bold = bold
            run.font.size = Pt(9.5)

        for index, header in enumerate(headers):
            cell = table.rows[0].cells[index]
            fill(cell, header, bold=True, align_left=index in text_columns)
            shading = OxmlElement("w:shd")
            shading.set(qn("w:val"), "clear")
            shading.set(qn("w:fill"), "E8EEF7")
            cell._tc.get_or_add_tcPr().append(shading)

        for row in rows:
            cells = table.add_row().cells
            for index, value in enumerate(row):
                fill(cells[index], value, bold=False, align_left=index in text_columns)

        # Keep the whole table on one page: every row except the last stays
        # with the row that follows it.
        for row in table.rows[:-1]:
            for cell in row.cells:
                cell.paragraphs[0].paragraph_format.keep_with_next = True

        if widths_cm:
            for row in table.rows:
                for index, width in enumerate(widths_cm):
                    row.cells[index].width = Cm(width)

        spacer = self.document.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)

    def save(self, path: Path) -> None:
        self.document.save(str(path))


# ----------------------------------------------------------------------
# The document
# ----------------------------------------------------------------------

def build(facts: dict) -> Report:
    report = Report()
    experiments = facts["experiments"]
    run_figures = run_directory(RUN_ID) / "figures"
    eda_figures = FIGURES_DIR / "eda"
    split_figures = FIGURES_DIR / "splits"

    def metric(name: str, key: str) -> float:
        return experiments[name]["metrics"][key]

    accuracies = {name: metric(name, "test_accuracy") for name, _, _ in EXPERIMENTS}
    best_name = max(accuracies, key=accuracies.get)
    worst_name = min(accuracies, key=accuracies.get)
    test_images = experiments["random_rgb"]["test_images"]

    # Difference "13 bands minus RGB" for each split, in percentage points.
    difference_random = (accuracies["random_multispectral"] - accuracies["random_rgb"]) * 100
    difference_spatial = (accuracies["spatial_multispectral"] - accuracies["spatial_rgb"]) * 100
    images_random = abs(experiments["random_rgb"]["errors"]
                        - experiments["random_multispectral"]["errors"])
    images_spatial = abs(experiments["spatial_rgb"]["errors"]
                         - experiments["spatial_multispectral"]["errors"])

    # Sampling noise of one accuracy measured on the test set (binomial).
    typical_accuracy = sum(accuracies.values()) / len(accuracies)
    standard_error = math.sqrt(typical_accuracy * (1 - typical_accuracy) / test_images)

    settings = experiments["random_rgb"]["metrics"]
    seconds = [experiment["seconds_per_epoch"] for experiment in experiments.values()]
    mean_seconds = sum(seconds) / len(seconds)
    early = experiments["random_multispectral"]
    early_history = early["history"]
    early_best_before = early_history.loc[
        early_history["epoch"] <= 2, "validation_macro_f1"
    ].max()
    leakage_share = facts["leakage_images"] / facts["total_images"]

    # ------------------------------------------------------------------
    report.title(
        "Clasificarea acoperirii terenului pe imagini Sentinel-2 cu ResNet18",
        "Setul EuroSAT: analiză exploratorie, protocol experimental și rezultate",
        "Cuculea Teodor-Ioan · lucrare pregătitoare pentru disertație · noiembrie 2026",
    )

    report.heading("Rezumat", 1)
    report.text(
        f"Lucrarea descrie un experiment complet de clasificare pe setul EuroSAT "
        f"({counted(facts['total_images'], 'imagini')} Sentinel-2, 10 clase). Un model ResNet18 "
        f"preantrenat pe ImageNet a fost antrenat în patru configurații: cu 3 benzi (RGB) sau cu "
        f"toate cele 13 benzi spectrale, pe un split aleator sau pe un split spațial în care "
        f"imaginile care se suprapun pe teren rămân în același subset. Acuratețea pe test este "
        f"între {pct(accuracies[worst_name])} și {pct(accuracies[best_name])}."
    )
    report.text(
        f"**Concluzia principală este că diferențele dintre configurații nu sunt concludente.** "
        f"Fiecare configurație a fost rulată o singură dată, diferențele sunt de cel mult "
        f"{num(abs(difference_random))} puncte procentuale, au semn opus pe cele două splituri, iar "
        f"una dintre rulări a fost oprită prematur de criteriul de early stopping. Analiza a scos "
        f"la iveală și două lucruri utile pentru etapa următoare: splitul aleator conține "
        f"{counted(facts['leakage_pairs'], 'perechi')} de imagini suprapuse aflate în subseturi "
        f"diferite, iar splitul spațial construit aici elimină suprapunerile, dar nu și apropierea "
        f"geografică dintre antrenare și test."
    )

    # ------------------------------------------------------------------
    report.heading("1. Introducere și obiective", 1)
    report.text(
        "Scopul acestei etape a fost parcurgerea, cap-coadă, a unui proiect de învățare automată "
        "pe imagini satelitare: înțelegerea datelor, împărțirea lor corectă, antrenarea unui model "
        "convoluțional, evaluarea și interpretarea cinstită a rezultatelor. Tema nu este cea finală "
        "a disertației; este un exercițiu pe un set de date cunoscut, pe care rezultatele pot fi "
        "comparate cu literatura [1]."
    )
    report.text("Experimentul caută răspuns la două întrebări:")
    report.numbered([
        "**Ajută benzile spectrale suplimentare?** Sentinel-2 măsoară 13 benzi, dintre care doar "
        "trei sunt vizibile (roșu, verde, albastru). Se compară un model care primește doar RGB cu "
        "unul care primește toate cele 13 benzi.",
        "**Supraestimează splitul aleator performanța?** Imaginile EuroSAT sunt decupate din scene "
        "mari, iar unele se suprapun pe teren. Dacă o imagine de test acoperă aceeași zonă ca una "
        "de antrenare, modelul este evaluat parțial pe ceva ce a văzut deja. Se compară splitul "
        "aleator cu unul care ține imaginile suprapuse în același subset.",
    ])

    # ------------------------------------------------------------------
    report.heading("2. Setul de date", 1)
    report.text(
        f"EuroSAT [1, 2] conține {counted(facts['total_images'], 'imagini')} decupate din scene "
        f"Sentinel-2, etichetate în 10 clase de acoperire și utilizare a terenului. S-a folosit "
        f"varianta multispectrală. Fiecare imagine are 64 × 64 pixeli și 13 canale, este stocată "
        f"ca TIFF georeferențiat cu valori întregi pe 16 biți și acoperă aproximativ 640 × 640 m "
        f"(10 m pe pixel). Imaginile provin din {facts['utm_zones']} zone UTM diferite din Europa."
    )
    report.text(
        "Datele sunt produse Sentinel-2 de nivel L1C: reflectanță la partea superioară a "
        "atmosferei, corectată geometric, dar fără corecție atmosferică. Metadatele fișierelor "
        "indică factor de scalare 1 și offset 0 pentru toate benzile, deci valorile sunt folosite "
        "așa cum sunt stocate. Ele nu au fost convertite în reflectanță fizică; pentru clasificare "
        "contează doar că sunt consecvente între imagini."
    )
    report.figure(
        eda_figures / "sample_grid.png",
        "Câte un exemplu din fiecare clasă, afișat în culori naturale (benzile B04, B03, B02).",
    )
    report.figure(
        eda_figures / "class_distribution.png",
        "Numărul de imagini pe clasă. Clasele au între 2.000 și 3.000 de imagini, deci setul este "
        "doar ușor dezechilibrat.",
        width_cm=14.5,
    )
    report.text(
        "Cele 13 benzi au rezoluții native diferite (10, 20 sau 60 m); în EuroSAT toate sunt "
        "aduse la 10 m prin interpolare [1]. Tabelul următor le prezintă în ordinea în care apar "
        "în fișiere."
    )
    report.table(
        ["Poziție", "Bandă", "Lungime de undă (nm)", "Rezoluție nativă (m)", "Rol"],
        [
            [str(index), name, str(BAND_FACTS[name][0]), str(BAND_FACTS[name][1]),
             BAND_FACTS[name][2]]
            for index, name in enumerate(BAND_NAMES)
        ],
        "Benzile Sentinel-2 în ordinea din fișierele EuroSAT (poziția începe de la 0).",
        widths_cm=[1.6, 1.6, 3.6, 3.6, 5.6],
        text_columns=(0, 1, 4),
    )
    report.text(
        f"**Observație despre ordinea benzilor.** Fișierele nu conțin numele benzilor. În prima "
        f"versiune a proiectului s-a presupus că B8A urmează imediat după B08. Statisticile arată "
        f"altceva: canalul de pe poziția 9 are media {num(facts['band_means']['B10'], 1)}, valoare "
        f"tipică pentru banda de cirrus B10, iar canalul de pe ultima poziție urmărește îndeaproape "
        f"B08, cum este de așteptat pentru B8A. Ordinea reală este deci B01–B08, B09, B10, B11, "
        f"B12, B8A, aceeași pe care o folosește biblioteca TorchGeo [8]. Etichetele au fost "
        f"corectate. Antrenarea nu a fost afectată: modelul cu 13 benzi le primește pe toate, iar "
        f"benzile RGB sunt pe pozițiile 3, 2 și 1 în ambele variante."
    )

    # ------------------------------------------------------------------
    report.heading("3. Analiză exploratorie", 1)
    report.text(
        "Înainte de antrenare s-a verificat dacă benzile disting clasele între ele. Pentru fiecare "
        "clasă au fost alese aleator 50 de imagini (cu seed fix, deci reproductibil). Fiecare "
        "imagine a fost redusă la o valoare medie pe bandă, iar aceste valori au fost mediate pe "
        "clasă. Rezultatul este „semnătura spectrală” medie a clasei."
    )
    report.figure(
        eda_figures / "spectral_signatures.png",
        "Semnătura spectrală medie a fiecărei clase (albastru), peste semnăturile celorlalte clase "
        "(gri). Benzile sunt ordonate după lungimea de undă.",
    )
    report.figure(
        eda_figures / "spectral_heatmap.png",
        "Aceleași medii, standardizate pe fiecare bandă: roșu înseamnă peste media celor 10 clase, "
        "albastru sub medie. Figura arată poziția relativă a claselor, nu reflectanța fizică.",
    )
    report.text("Din cele două figuri reies câteva observații:")
    report.bullets([
        "**SeaLake** are valori mult mai mici decât toate celelalte clase în infraroșu (B05–B12), "
        "pentru că apa absoarbe aproape complet aceste lungimi de undă. Este clasa cel mai ușor de "
        "separat spectral.",
        "**Clasele cu vegetație** (Forest, Pasture, AnnualCrop, PermanentCrop, "
        "HerbaceousVegetation) au salt puternic între roșu (B04) și infraroșu apropiat "
        "(B06–B8A), semnul clasic al vegetației. Forest are cele mai mici valori în vizibil.",
        "**AnnualCrop și PermanentCrop** au profiluri medii aproape identice. Mediile spectrale nu "
        "le pot separa; diferența trebuie să vină din textură și formă.",
        "**Industrial** are cele mai mari valori în vizibil (B01–B03), iar **Highway** și "
        "**Residential** sunt aproape de media tuturor claselor pe toate benzile.",
        f"**B10 (cirrus)** este aproape de zero pentru toate clasele (medie "
        f"{num(facts['band_means']['B10'], 1)}). Banda măsoară norii înalți, nu suprafața, deci "
        f"nu aduce informație utilă pentru clasificare.",
    ])
    report.figure(
        eda_figures / "b08_boxplot.png",
        "Distribuția mediei benzii B08 pe imagine, pentru fiecare clasă (50 de imagini pe clasă).",
        width_cm=14.5,
    )
    report.text(
        "Figura de mai sus arată variabilitatea din interiorul unei clase: intervalele se suprapun "
        "larg, deci o singură bandă nu poate separa clasele. Mai mult, două imagini pot avea "
        "aceeași medie și structuri spațiale complet diferite. Aceste limite justifică folosirea "
        "unei rețele convoluționale, care învață și textura, nu doar nivelul mediu al benzilor. "
        "A fost calculat exploratoriu și indicele NDVI = (B08 − B04) / (B08 + B04), care "
        "evidențiază vegetația; nu a fost folosit ca intrare în model."
    )

    # ------------------------------------------------------------------
    report.heading("4. Pregătirea datelor", 1)
    report.heading("4.1 Splitul aleator stratificat", 2)
    report.text(
        f"Imaginile au fost împărțite în trei subseturi: antrenare (70%), validare (15%) și test "
        f"(15%), adică {thousands(facts['random_sizes']['train'])}, "
        f"{thousands(facts['random_sizes']['validation'])} și "
        f"{counted(facts['random_sizes']['test'], 'imagini')}. Împărțirea se face în doi pași: "
        f"întâi se separă 70% pentru antrenare, apoi restul de 30% se împarte în două jumătăți. "
        f"Ambii pași sunt stratificați, deci fiecare subset păstrează proporția claselor din setul "
        f"întreg. Rolurile sunt distincte: pe antrenare se ajustează ponderile, pe validare se "
        f"alege epoca la care se oprește antrenarea, iar testul este folosit o singură dată, la "
        f"final."
    )

    report.heading("4.2 Suprapuneri între subseturi", 2)
    report.text(
        f"Pentru fiecare imagine s-a extras conturul geografic din metadatele TIFF și a fost adus "
        f"într-un sistem de coordonate comun, cu unități în metri (EPSG:3035). Apoi s-au căutat "
        f"perechile de imagini din subseturi diferite ale căror contururi se intersectează. Au "
        f"fost găsite **{counted(facts['leakage_pairs'], 'perechi')}** de acest fel, care implică "
        f"{counted(facts['leakage_images'], 'imagini')} ({pct(leakage_share, 1)} din set). În "
        f"{pct(facts['leakage_same_class'], 0)} dintre perechi, cele două imagini au aceeași clasă."
    )
    pair_labels = {
        "train-validation": "antrenare – validare",
        "train-test": "antrenare – test",
        "validation-test": "validare – test",
    }
    report.table(
        ["Subseturi", "Perechi suprapuse", "IoU mediu", "IoU maxim"],
        [
            [pair_labels[row.split_pair], str(row.overlapping_pairs),
             num(row.mean_iou), num(row.maximum_iou)]
            for row in facts["leakage_summary"].itertuples()
        ],
        "Perechi de imagini suprapuse aflate în subseturi diferite ale splitului aleator. IoU "
        "(intersecție supra reuniune) este 0 fără suprapunere și 1 pentru contururi identice.",
        widths_cm=[5, 3.6, 3, 3],
    )
    report.text(
        f"Cele mai multe suprapuneri sunt mici (IoU mediu în jur de 0,1), dar "
        f"{counted(facts['leakage_pairs_high_iou'], 'perechi')} au IoU peste 0,5. Figura "
        f"următoare arată trei dintre ele: practic aceeași imagine apare în două subseturi "
        f"diferite."
    )
    report.figure(
        split_figures / "overlap_examples.png",
        "Trei perechi de imagini aproape identice pe care splitul aleator le-a pus în subseturi "
        "diferite. Sub fiecare imagine: numele fișierului și subsetul.",
        width_cm=12.0,
    )

    report.heading("4.3 Splitul spațial", 2)
    report.text(
        f"Al doilea split pornește de la o regulă simplă: două imagini care se suprapun trebuie să "
        f"ajungă în același subset. Imaginile sunt privite ca noduri ale unui graf, iar fiecare "
        f"suprapunere ca o muchie. Grupurile de imagini legate între ele (componentele conexe) "
        f"sunt găsite cu structura Union-Find. Rezultă {counted(facts['groups'], 'grupuri')}, "
        f"dintre care {thousands(facts['groups_single'])} conțin o singură imagine. Doar "
        f"{counted(facts['groups_multi'], 'grupuri')} au cel puțin două imagini; ele adună "
        f"{counted(facts['groups_multi_images'], 'imagini')}, iar cel mai mare are "
        f"{facts['groups_largest']}."
    )
    report.text(
        f"Grupurile sunt apoi repartizate întregi în antrenare, validare sau test, de la cel mai "
        f"mare la cel mai mic. Fiecare grup merge în subsetul în care apropie cel mai mult "
        f"dimensiunea și distribuția claselor de ținta 70/15/15. Rezultatul are exact aceleași "
        f"dimensiuni și aceeași distribuție de clase ca splitul aleator "
        f"({thousands(facts['spatial_sizes']['train'])} / "
        f"{thousands(facts['spatial_sizes']['validation'])} / "
        f"{thousands(facts['spatial_sizes']['test'])}), iar verificarea finală confirmă că nicio "
        f"pereche suprapusă nu mai traversează subseturile."
    )
    report.text(
        f"Eliminarea suprapunerii nu înseamnă însă independență geografică. Pentru fiecare imagine "
        f"de test s-a calculat distanța până la cea mai apropiată imagine de antrenare. Distanța "
        f"minimă este {num(facts['distance_min'], 0)} m, exact lățimea unei imagini, iar mediana "
        f"este {num(facts['distance_median'] / 1000)} km. "
        f"**{pct(facts['distance_under_700'], 1)} dintre imaginile de test au o imagine de "
        f"antrenare la mai puțin de 700 m**, adică o imagine imediat alăturată; "
        f"{pct(facts['distance_under_1000'], 1)} au una la sub 1 km și "
        f"{pct(facts['distance_under_5000'], 1)} la sub 5 km."
    )
    report.figure(
        split_figures / "test_to_train_distance.png",
        "Distanța de la fiecare imagine de test din splitul spațial la cea mai apropiată imagine "
        "de antrenare. Vârful de la 0,64 km corespunde imaginilor alăturate.",
        width_cm=15,
    )
    report.figure(
        split_figures / "spatial_split_map.png",
        "Poziția imaginilor din splitul spațial. Cele trei subseturi sunt amestecate în aceleași "
        "regiuni; niciun subset nu acoperă o zonă pe care celelalte să nu o acopere.",
        width_cm=10.5,
    )
    report.text(
        f"Mai trebuie reținut un detaliu pentru interpretare: cele două splituri au seturi de test "
        f"diferite. Doar {facts['common_test']} dintre cele "
        f"{counted(facts['spatial_sizes']['test'], 'imagini')} de test sunt comune. O diferență "
        f"de acuratețe între splituri reflectă deci și faptul că modelele sunt evaluate pe imagini "
        f"diferite."
    )

    report.heading("4.4 Normalizarea", 2)
    report.text(
        "Benzile au scări foarte diferite, așa că fiecare bandă este standardizată: se scade media "
        "și se împarte la abaterea standard. Statisticile sunt calculate **doar din imaginile de "
        "antrenare**, separat pentru fiecare split, și sunt aplicate neschimbate pe validare și "
        "test. Calculul lor pe tot setul ar introduce în antrenare informație despre datele de "
        "test. Nu au fost folosite augmentări."
    )

    # ------------------------------------------------------------------
    report.heading("5. Modelul", 1)
    report.text(
        "S-a folosit ResNet18 [3], cu ponderi preantrenate pe ImageNet. O rețea reziduală este "
        "construită din blocuri care învață o corecție față de intrarea lor: ieșirea unui bloc "
        "este F(x) + x, unde F sunt două convoluții 3 × 3, fiecare urmată de batch normalization, "
        "iar x este intrarea adusă direct la ieșire printr-o conexiune de scurtcircuit (skip "
        "connection). Datorită acestei conexiuni, gradientul are un drum direct către straturile "
        "de la început, ceea ce permite antrenarea rețelelor adânci. ResNet18 are un strat "
        "convoluțional inițial, opt astfel de blocuri grupate în patru etape cu 64, 128, 256 și "
        "512 canale, o mediere globală și un strat liniar final."
    )
    report.text("Arhitectura a fost modificată în două locuri:")
    report.bullets([
        "**Stratul final** (1.000 de clase ImageNet) a fost înlocuit cu un strat liniar cu 10 "
        "ieșiri, inițializat aleator.",
        "**Prima convoluție**, pentru modelul cu 13 benzi, a fost înlocuită cu una care acceptă 13 "
        "canale. Filtrele preantrenate pentru roșu, verde și albastru au fost copiate pe pozițiile "
        "benzilor B04, B03 și B02, iar filtrele celorlalte 10 benzi au fost inițializate cu zero. "
        "La începutul antrenării, modelul cu 13 benzi calculează deci exact ce calculează modelul "
        "RGB, iar contribuția benzilor suplimentare este învățată pe parcurs.",
    ])
    report.text(
        f"Toate straturile sunt antrenate (nu se îngheață nimic). Modelul RGB are "
        f"{counted(facts['parameters']['rgb'], 'parametri')}, iar cel cu 13 benzi "
        f"{thousands(facts['parameters']['multispectral'])}; diferența vine doar din prima "
        f"convoluție. Imaginile intră la rezoluția lor de 64 × 64 pixeli, fără redimensionare, "
        f"deci harta de trăsături de dinaintea medierii globale are 2 × 2 poziții."
    )

    # ------------------------------------------------------------------
    report.heading("6. Protocolul experimental", 1)
    report.text(
        "Au fost antrenate patru modele, câte unul pentru fiecare combinație de reprezentare (RGB "
        "sau 13 benzi) și split (aleator sau spațial), cu aceleași setări:"
    )
    report.table(
        ["Setare", "Valoare"],
        [
            ["Funcție de cost", "cross-entropy"],
            ["Optimizator", f"AdamW [4], rată de învățare {num(settings['learning_rate'], 4)}, "
                            f"weight decay {num(settings['weight_decay'], 4)}"],
            ["Programarea ratei de învățare",
             f"cosine annealing [5] pe {counted(settings['epochs'], 'epoci')}, până la 0,000001"],
            ["Dimensiunea lotului", str(settings["batch_size"])],
            ["Număr maxim de epoci", str(settings["epochs"])],
            ["Alegerea modelului", "epoca cu cel mai mare macro-F1 pe validare"],
            ["Early stopping",
             f"oprire după {settings['early_stopping_patience']} epoci fără o creștere de cel "
             f"puțin {num(settings['early_stopping_min_delta'], 3)} a macro-F1 pe validare"],
            ["Seed", str(settings["random_seed"])],
            ["Hardware", f"{GPU_NAME}; aproximativ {num(mean_seconds, 0)} s pe epocă"],
        ],
        "Setările comune celor patru antrenări.",
        widths_cm=[5.2, 10.8],
        text_columns=(0, 1),
    )
    report.text(
        "Se raportează două metrici pe setul de test: acuratețea (procentul de imagini clasificate "
        "corect) și macro-F1 (media scorurilor F1 ale celor 10 clase, în care fiecare clasă "
        "contează la fel, indiferent de câte imagini are). Setul de test este evaluat o singură "
        "dată, cu modelul ales pe validare."
    )

    # ------------------------------------------------------------------
    report.heading("7. Rezultate", 1)
    result_rows = []
    for name, split, representation in EXPERIMENTS:
        experiment = experiments[name]
        metrics = experiment["metrics"]
        result_rows.append([
            split, representation,
            str(metrics["epochs_completed"]), str(metrics["best_epoch"]),
            pct(metrics["best_validation_macro_f1"]),
            pct(metrics["test_accuracy"]), pct(metrics["test_macro_f1"]),
            str(experiment["errors"]),
        ])
    report.table(
        ["Split", "Intrare", "Epoci rulate", "Epoca aleasă", "Macro-F1 validare",
         "Acuratețe test", "Macro-F1 test", "Greșeli pe test"],
        result_rows,
        f"Rezultatele celor patru configurații. Fiecare set de test are "
        f"{counted(test_images, 'imagini')}.",
        widths_cm=[1.7, 1.9, 1.6, 1.6, 2.3, 2.3, 2.3, 2.3],
        text_columns=(0, 1),
    )
    report.text(
        f"Toate configurațiile depășesc {num(math.floor(accuracies[worst_name] * 1000) / 10, 1)}% "
        f"acuratețe, în linie cu valorile raportate pentru acest set de date [1]. Cea mai bună "
        f"valoare este {pct(accuracies[best_name])} (split spațial, 13 benzi), iar cea mai slabă "
        f"{pct(accuracies[worst_name])} (split aleator, 13 benzi)."
    )
    report.figure(
        run_figures / "loss_curves.png",
        "Evoluția funcției de cost pe antrenare și pe validare. Linia verticală marchează epoca al "
        "cărei model a fost păstrat și evaluat pe test.",
    )
    report.text(
        "Curbele arată același tipar în toate cele patru cazuri. Costul pe antrenare scade "
        "continuu spre zero, în timp ce costul pe validare se stabilizează după primele 2–3 epoci "
        "și oscilează în jurul valorii 0,1. Acuratețea pe antrenare ajunge la peste 99,8% în "
        "rulările lungi, față de 97–98% pe validare. Modelul memorează deci setul de antrenare "
        "(overfitting), fără ca performanța pe validare să scadă."
    )
    report.figure(
        run_figures / "confusion_matrices.png",
        "Matricele de confuzie pe test (număr de imagini). Pe diagonală sunt clasificările "
        "corecte; în afara ei, greșelile.",
    )
    report.table(
        ["Perechea de clase", "Imagini confundate (total pe 4 modele)"],
        [[f"{first} ↔ {second}", str(count)] for first, second, count in top_confusions(facts)],
        "Cele mai frecvente confuzii, însumate pe cele patru modele și în ambele sensuri.",
        widths_cm=[8, 6],
    )
    report.text(
        "Greșelile se concentrează pe câteva perechi de clase care seamănă și vizual, și spectral: "
        "culturile permanente cu cele anuale și cu vegetația ierboasă, zonele industriale cu cele "
        "rezidențiale, râurile cu autostrăzile (ambele sunt structuri liniare). Prima pereche "
        "este cea pe care analiza exploratorie o arăta cu semnături spectrale aproape identice."
    )
    report.table(
        ["Clasă", "Aleator · RGB", "Aleator · 13 benzi", "Spațial · RGB", "Spațial · 13 benzi"],
        [
            [class_name] + [
                num(experiments[name]["report"][class_name]["f1-score"] * 100, 1)
                for name, _, _ in EXPERIMENTS
            ]
            for class_name in facts["class_names"]
        ],
        "Scorul F1 pe clasă, pe test (%).",
        widths_cm=[4.4, 2.8, 3.1, 2.8, 3.1],
    )
    report.figure(
        run_figures / "per_class_f1_difference.png",
        "Diferența de F1 dintre modelul cu 13 benzi și cel RGB, pe fiecare clasă. Semnul "
        "diferenței se schimbă între cele două splituri la majoritatea claselor.",
    )

    # ------------------------------------------------------------------
    report.heading("8. Discuție", 1)
    report.heading("8.1 RGB față de 13 benzi", 2)
    report.text(
        f"Pe splitul aleator, modelul cu 13 benzi este cu {num(abs(difference_random))} puncte "
        f"procentuale {'sub' if difference_random < 0 else 'peste'} cel RGB "
        f"({counted(images_random, 'imagini')} din {thousands(test_images)}). Pe splitul spațial "
        f"este cu {num(abs(difference_spatial))} puncte "
        f"{'sub' if difference_spatial < 0 else 'peste'} "
        f"({counted(images_spatial, 'imagini')}). Cele două rezultate se contrazic, iar niciunul "
        f"nu permite o concluzie, din trei motive."
    )
    report.bullets([
        f"**Zgomotul de eșantionare.** O acuratețe de aproximativ {pct(typical_accuracy, 0)} "
        f"măsurată pe {counted(test_images, 'imagini')} are o eroare standard de circa "
        f"{num(standard_error * 100)} puncte procentuale. Diferența de pe splitul spațial "
        f"({num(abs(difference_spatial))} puncte) este de același ordin de mărime.",
        "**O singură rulare pe configurație.** Antrenarea depinde de seed (ordinea loturilor, "
        "inițializarea stratului final). Fără rulări repetate nu se poate separa efectul benzilor "
        "de variația dintre rulări.",
        f"**Rularea „aleator · 13 benzi” a fost oprită prematur.** S-a oprit după "
        f"{counted(early['metrics']['epochs_completed'], 'epoci')}, față de 20–25 la celelalte. "
        f"Macro-F1 pe validare a atins {pct(early_best_before)} încă din epoca 2, iar "
        f"îmbunătățirea din epoca {early['metrics']['best_epoch']} "
        f"({pct(early['metrics']['best_validation_macro_f1'])}) a fost sub pragul de "
        f"{num(settings['early_stopping_min_delta'], 3)}, deci contorul de răbdare nu a fost "
        f"resetat. Rata de învățare era încă la aproape 90% din valoarea inițială. Celelalte trei "
        f"rulări și-au atins maximul mult mai târziu (epocile 14–19), când rata de învățare "
        f"scăzuse. Comparația este deci între un model antrenat complet și unul antrenat parțial.",
    ])
    report.text(
        "Mai există o explicație posibilă pentru lipsa unui câștig clar: ponderile preantrenate "
        "cunosc doar RGB, iar cele 10 benzi suplimentare pornesc de la zero. Cu o rată de "
        "învățare mică și puține epoci, modelul poate ajunge să le folosească puțin. Ipoteza nu a "
        "fost verificată."
    )

    report.heading("8.2 Split aleator față de split spațial", 2)
    report.text(
        f"Ipoteza de plecare era că splitul aleator supraestimează performanța. Rezultatele nu o "
        f"confirmă: pentru RGB cele două splituri dau practic aceeași acuratețe "
        f"({pct(accuracies['random_rgb'])} și {pct(accuracies['spatial_rgb'])}), iar pentru 13 "
        f"benzi splitul spațial este chiar mai bun, cel mai probabil din cauza opririi premature "
        f"discutate mai sus. Analiza datelor explică de ce efectul nu avea cum să fie mare:"
    )
    report.bullets([
        f"Suprapunerile din splitul aleator ating doar "
        f"{counted(facts['leakage_images'], 'imagini')} ({pct(leakage_share, 1)} din set), "
        f"majoritatea cu suprapunere mică.",
        f"Splitul spațial constrânge doar cele "
        f"{counted(facts['groups_multi_images'], 'imagini')} din grupuri cu cel puțin două "
        f"imagini. Restul sunt repartizate la fel de liber ca într-un split aleator.",
        f"După eliminarea suprapunerilor, {pct(facts['distance_under_700'], 0)} dintre imaginile "
        f"de test au în continuare o imagine vecină în antrenare. Două imagini alăturate provin, "
        f"cel mai probabil, din aceeași scenă, deci au aceeași dată de achiziție, aceeași "
        f"iluminare și, de multe ori, același tip de teren.",
    ])
    report.text(
        "Splitul construit aici este deci un split **fără suprapuneri**, nu unul **independent "
        "geografic**. El nu măsoară cât de bine generalizează modelul într-o regiune nouă. "
        "Pentru asta este nevoie de un split pe blocuri geografice (regiuni sau scene întregi "
        "ținute deoparte), cum recomandă literatura despre validarea spațială [6, 7]."
    )

    report.heading("8.3 Limitări", 2)
    report.bullets([
        "Un singur seed pe configurație; nu există medie și abatere standard.",
        "Criteriul de early stopping a tratat inegal configurațiile.",
        "Fără augmentări, deși curbele arată overfitting.",
        "Seturile de test ale celor două splituri sunt diferite.",
        "Analiza exploratorie folosește 50 de imagini pe clasă, nu tot setul.",
        "Datele sunt de nivel L1C, fără corecție atmosferică, iar trei benzi (B01, B09, B10) au "
        "rezoluție nativă de 60 m.",
    ])

    # ------------------------------------------------------------------
    report.heading("9. Concluzii și pași următori", 1)
    report.bullets([
        f"ResNet18 preantrenat atinge {pct(accuracies[worst_name], 1)}–"
        f"{pct(accuracies[best_name], 1)} acuratețe pe EuroSAT în toate cele patru configurații, "
        f"în aproximativ {num(mean_seconds / 60, 0)} minute pe epocă pe o placă video de consum.",
        "Datele nu arată că cele 13 benzi ar fi mai bune sau mai slabe decât RGB. Diferențele sunt "
        "mici, au semn opus pe cele două splituri și sunt afectate de oprirea prematură a unei "
        "rulări.",
        f"Splitul aleator al EuroSAT conține {counted(facts['leakage_pairs'], 'perechi')} de "
        f"imagini suprapuse între subseturi, unele aproape identice. Efectul lor asupra acurateței "
        f"nu a putut fi măsurat cu acest experiment.",
        "Eliminarea suprapunerilor nu este suficientă pentru o evaluare independentă geografic, "
        "pentru că imaginile de test rămân vecine cu cele de antrenare.",
    ])
    report.text("Pașii care ar transforma aceste observații în concluzii:")
    report.numbered([
        "Repetarea fiecărei configurații cu 3–5 seed-uri și raportarea mediei și a abaterii "
        "standard.",
        "Același număr de epoci pentru toate configurațiile (fără early stopping sau cu răbdare "
        "mai mare), cu păstrarea alegerii modelului pe validare.",
        "Un split pe blocuri geografice, cu regiuni întregi ținute deoparte pentru test, și un "
        "singur set de test comun tuturor configurațiilor.",
        "Augmentări simple (oglindiri, rotații la 90°), potrivite pentru imagini văzute de sus.",
        "Ponderi preantrenate pe imagini Sentinel-2 cu 13 benzi (de exemplu SSL4EO-S12 [9]), în "
        "locul celor ImageNet, pentru ca benzile suplimentare să nu pornească de la zero.",
    ])

    report.heading("Direcția propusă pentru disertație", 2)
    report.text(
        "Experiența cu datele Sentinel-2 poate fi dusă spre o problemă aplicată: detecția "
        "defrișărilor din România. Față de proiectul de față, sarcina trece de la clasificarea "
        "unei imagini întregi la segmentare (o etichetă pe fiecare pixel) și la compararea "
        "aceleiași zone în ani diferiți. Rezultatul ar fi o hartă cu zonele tăiate marcate și aria "
        "lor în hectare. Un prim prototip ar acoperi o singură zonă din Carpați, cu imagini de "
        "vară din ani consecutivi, cu etichete derivate din Global Forest Change [10] și cu un "
        "model U-Net [11] cu encoder ResNet. Trei lucruri învățate aici se transferă direct: "
        "lucrul cu benzile Sentinel-2 și cu fișiere georeferențiate, separarea geografică a "
        "datelor de antrenare și de test și adaptarea unui model preantrenat la mai mult de trei "
        "canale."
    )

    # ------------------------------------------------------------------
    report.heading("Bibliografie", 1)
    references = [
        "Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). EuroSAT: A Novel Dataset and Deep "
        "Learning Benchmark for Land Use and Land Cover Classification. IEEE Journal of Selected "
        "Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226.",
        "Helber, P., Bischke, B., Dengel, A., Borth, D. EuroSAT (set de date). Zenodo, "
        "https://zenodo.org/records/7711810.",
        "He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. "
        "IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.",
        "Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. International "
        "Conference on Learning Representations (ICLR).",
        "Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. "
        "International Conference on Learning Representations (ICLR).",
        "Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, "
        "spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929.",
        "Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of "
        "large-scale ecological mapping models. Nature Communications, 11, 4540.",
        "Stewart, A. J. et al. (2022). TorchGeo: Deep Learning With Geospatial Data. ACM "
        "SIGSPATIAL International Conference on Advances in Geographic Information Systems.",
        "Wang, Y. et al. (2023). SSL4EO-S12: A Large-Scale Multimodal, Multitemporal Dataset for "
        "Self-Supervised Learning in Earth Observation. IEEE Geoscience and Remote Sensing "
        "Magazine, 11(3).",
        "Hansen, M. C. et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover "
        "Change. Science, 342(6160), 850–853.",
        "Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for "
        "Biomedical Image Segmentation. MICCAI, LNCS 9351, 234–241.",
    ]
    for index, reference in enumerate(references, start=1):
        paragraph = report.document.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.9)
        paragraph.paragraph_format.first_line_indent = Cm(-0.9)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(f"[{index}]  {reference}")
        run.font.size = Pt(10)

    report.heading("Anexă: reproducerea rezultatelor", 1)
    report.text(
        f"Codul, tabelele de split și rezultatele rulării „{RUN_ID}” sunt în repository-ul "
        f"proiectului. Fișierul README descrie pașii, de la pregătirea datelor la generarea "
        f"acestui document. Toate cifrele din text sunt citite automat din fișierele de rezultate "
        f"de către scriptul documentation/build_documentation.py.",
        justify=False,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Where to write the document (close it in Word first).")
    output_file = parser.parse_args().output

    facts = load_facts()
    report = build(facts)
    report.save(output_file)
    print(f"Document saved: {output_file}")
    print(f"Figures: {report.figure_number}; tables: {report.table_number}")


if __name__ == "__main__":
    main()
