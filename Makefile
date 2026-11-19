PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
NBS    := notebooks/TT2_Airline_ejecutado.ipynb notebooks/TT2_Diabetes_ejecutado.ipynb notebooks/TT2_Polish_Bankruptcy_ejecutado.ipynb

.PHONY: setup data notebooks tablero dashboard clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

data:
	$(BIN)/python scripts/download_data.py

# Re-ejecuta los tres notebooks de experimentación (escriben artefactos en ./artifacts/).
notebooks:
	@for nb in $(NBS); do \
		echo "Ejecutando $$nb"; \
		$(BIN)/jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

# Prepara datos_tablero/ para la vista ejecutiva (usa runs/ y data/).
tablero:
	$(BIN)/python scripts/preparar_tablero.py

dashboard:
	$(BIN)/streamlit run app.py

clean:
	rm -rf $(VENV) data/airline data/diabetes data/polish datos_tablero .cache
