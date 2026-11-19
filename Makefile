PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup data notebooks dashboard all clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

data:
	$(BIN)/python scripts/download_data.py

notebooks:
	@for nb in notebooks/*.ipynb; do \
		echo "Ejecutando $$nb"; \
		$(BIN)/jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

dashboard:
	$(BIN)/streamlit run app.py

all: setup data notebooks

clean:
	rm -rf $(VENV) data/raw data/processed
