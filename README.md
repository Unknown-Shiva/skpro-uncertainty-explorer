# skpro Uncertainty Explorer

A web application built with **FastAPI** and **skpro** that accepts a CSV dataset and returns probabilistic regression predictions with uncertainty bands — visualized interactively in the browser.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![skpro](https://img.shields.io/badge/skpro-2.0+-orange)

---

## What It Does

Upload any numeric CSV dataset → the app:

1. Trains a **standard Ridge regressor** (point predictions)
2. Trains a **skpro `ResidualDouble`** probabilistic regressor
3. Returns **80% and 95% prediction intervals** for every test sample
4. Evaluates using **CRPS** and **Pinball Loss** (proper scoring rules)
5. Checks **interval calibration** (coverage vs nominal level)
6. Visualizes everything in an interactive chart

---

## Demo

```
Actual house value:    $320,000
Standard prediction:   $305,000        ← just one number, no confidence info

Probabilistic output:
  Mean prediction:     $312,000
  Std deviation:       $44,000
  80% interval:        [$255,000 — $369,000]
  95% interval:        [$226,000 — $398,000]
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/skpro-uncertainty-explorer
cd skpro-uncertainty-explorer

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

# Open browser
http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Web UI |
| `POST` | `/analyze` | Upload CSV + get probabilistic results |
| `GET`  | `/sample-csv` | Load built-in California Housing sample |
| `GET`  | `/health` | Health check |
| `GET`  | `/docs`   | Auto-generated Swagger API docs |

### Example API call

```python
import requests

with open("my_data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze?target_column=price",
        files={"file": f}
    )

result = response.json()
print(result["metrics"])
# {'mae_standard': 0.512, 'mae_probabilistic': 0.498,
#  'crps': 0.341, 'coverage_80': 81.2, 'coverage_95': 94.6, ...}
```

---

## CSV Format

Any CSV with:
- Numeric feature columns
- One numeric target column
- At least 20 rows

Example:
```
age,income,rooms,price
34,55000,4,320000
28,42000,3,275000
...
```

Then set `target_column=price`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| ML | skpro `ResidualDouble`, scikit-learn `Ridge` |
| Metrics | CRPS, Pinball Loss (skpro) |
| Frontend | Vanilla JS + Chart.js |
| Serving | Uvicorn ASGI |

---

## Key Concepts Demonstrated

- **Probabilistic regression** vs standard regression
- **ResidualDouble**: fits a mean model + variance model separately
- **CRPS** (Continuous Ranked Probability Score): evaluates full distribution quality
- **Pinball Loss**: evaluates quantile calibration
- **Coverage check**: are 80% intervals actually covering ~80% of values?

---

## Related

- [skpro GitHub](https://github.com/sktime/skpro)
- [skpro Documentation](https://skpro.readthedocs.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [European Summer of Code 2026](https://www.esoc.dev)

---

## Author

Built as part of preparation for contributing to [skpro](https://github.com/sktime/skpro) via [ESoC 2026](https://www.esoc.dev).
