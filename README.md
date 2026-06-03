# FuzzyDiary

**An open-source framework for morphology-aware and traceable linguistic summarization of clinical time series.**

FuzzyDiary transforms a univariate time series and a YAML configuration file into a self-contained, auditable narrative report. The framework detects morphological events (peaks, valleys, sustained increases and decreases, plateaus, oscillations, and regime changes), evaluates fuzzy protoforms, aggregates the resulting summaries over user-defined periods, and generates an interactive HTML report in which every textual statement can be traced back to the original signal.

The framework is domain-independent and can be applied to any univariate signal for which linguistic variables, fuzzy quantifiers, qualifiers, and morphological events can be defined through a YAML configuration.

---

## Key Features

FuzzyDiary is intended for analysts and researchers who need to transform raw univariate measurements into **human-readable, traceable narratives** without relying on black-box models. Starting from a data file containing timestamp and value fields for a univariate time series, together with a YAML configuration file, the framework provides the following capabilities:

* **Protoform-based linguistic summarization** using configurable fuzzy linguistic variables, quantifiers, and contexts.
* **Morphological event detection** (peaks, valleys, sustained increases/decreases, plateaus, oscillations, and regime changes) applied to RDP-simplified time series.
* **Automatic or manual epsilon selection** for the RDP simplification process.
* **Data descriptions** generated over user-defined time periods.
* **Self-contained interactive HTML reports** in which every textual statement is linked to the corresponding segment of the original signal, ensuring full traceability.
* **YAML-driven configuration**, allowing adaptation to new signals or domains without modifying the source code.
* **Command-line interface** (`fuzzydiary`) and **Python API** for programmatic integration.
* **Domain-independent architecture** that works with any timestamped univariate signal.

---

## Installation Requirements

* **Python 3.10 or later**
* **pip 21 or later** (or any modern package manager supporting PEP 517 builds)
* Supported operating systems: Linux, macOS, and Windows
* A modern web browser for viewing the generated HTML reports

The following Python dependencies are installed automatically:

`numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-fuzzy`, `rdp`, `pydantic`, `PyYAML`, `plotly`, and `jinja2`.

---

## Cloning the Repository

```bash
git clone https://github.com/ASIATIC257UJA/fuzzydiary.git
cd fuzzydiary
```

---

## Installing Dependencies

Using a dedicated virtual environment is strongly recommended:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -e .
```

The editable installation (`-e`) makes the `fuzzydiary` command-line utility available and allows the package to be imported from any Python script.

To verify that the installation was successful:

```bash
fuzzydiary version
```

---

## Usage

### Overview

FuzzyDiary is operated through the **`fuzzydiary` command-line interface**. Each execution requires two inputs:

1. A **CSV file** containing a timestamp column and a numerical value column.
2. A **YAML configuration file** that defines the signal characteristics, linguistic variables, fuzzy quantifiers, contexts, and events of interest.

The CLI provides both a complete end-to-end command (`run`) and separate subcommands for each stage of the workflow, which can be useful for inspection and debugging.

### Step-by-Step Execution

```bash
# 1. Inspect the time series
fuzzydiary load --series path/to/series.csv --config path/to/config.yaml

# 2. Simplify the series using RDP
fuzzydiary simplify --series path/to/series.csv --config path/to/config.yaml --epsilon auto

# 3. Detect morphological events
fuzzydiary events --series path/to/series.csv --config path/to/config.yaml

# 4. Generate linguistic descriptions
fuzzydiary describe --series path/to/series.csv --config path/to/config.yaml

# 5. Execute the full workflow and generate the HTML report
fuzzydiary run \
    --series path/to/series.csv \
    --config path/to/config.yaml \
    --output ./report/index.html
```

Open `./report/index.html` in any modern web browser to explore the interactive report.

---

## Example Use Case

The repository includes a ready-to-run example based on **Continuous Glucose Monitoring (CGM)** data:

* **Dataset:** `example_data/data.csv` — CGM measurements sampled every five minutes.
* **Configuration:** `example_yaml/config_glucose.yaml` — defines the linguistic variable (`very_low`, `low`, `medium`, `high`, `very_high`), fuzzy quantifiers (`many`, `most`, `almost_all`), the `day_moment` context (`night`, `morning`, `afternoon`, `evening`), and the meal-related context (`breakfast`, `lunch`, `dinner`).

Run the complete workflow with:

```bash
fuzzydiary run \
    --series example_data/data.csv \
    --config example_yaml/config_glucose.yaml \
    --output ./report/index.html
```

The generated `report/index.html` contains the linguistic narratives describing the selected period, together with interactive visualizations in which every textual statement is linked to the corresponding segment of the original signal.

---

## License

This project is distributed under the **MIT License**. See the `LICENSE` file for the complete license text.
