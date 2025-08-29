VENV=venv
PY=source $(VENV)/bin/activate;

setup:
	python -m venv $(VENV); $(PY) pip install --upgrade pip; \
	$(PY) pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

ppo:
	$(PY) python -m src.algos.ppo_train --config configs/ppo.yaml

reinforce:
	$(PY) python -m src.algos.reinforce_train --config configs/reinforce.yaml

grpo:
	$(PY) python -m src.algos.grpo_train --config configs/grpo.yaml

eval:
	$(PY) python -m src.eval.generate_eval
