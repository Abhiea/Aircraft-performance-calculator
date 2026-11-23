# Aircraft Performance Calculator

A Streamlit app that estimates the performance, stability and wing loads of a small propeller aircraft.
The defaults are a Cessna 172S. Open the app's **How to use** page for a full guide, and the
**Equations** page for every equation, its variables and its source.

## Run

    pip install -r requirements.txt
    streamlit run app.py

## Files

| File | Purpose |
|---|---|
| `app.py` | The interface (Calculator, How to use and Equations pages) |
| `core.py` | All calculations, in imperial units, and the input definitions with their descriptions and data status |
| `units.py` | Imperial and SI display conversion |
| `labels.py` | Turns variable names into LaTeX for display |
| `plots.py` | The figures |
| `docs_pages.py` | The How to use and Equations pages |
| `verify.py` | Re-evaluates every equation with separate code and compares with the calculator (`python verify.py`) |
| `.streamlit/config.toml` | Dark theme, and file-change polling so edits reload in synced folders |

## Input data status

Each input is tagged in `core.PARAM_INFO` as Published, Derived, Estimate, Placeholder or Setting, and flagged when
the value should come from an XFLR5 analysis for a new design. The sidebar shows these as badges.
