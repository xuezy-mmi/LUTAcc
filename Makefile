PYTHON ?= .venv/bin/python

.PHONY: bootstrap test test-ml lint-rtl rtl-smoke chipyard-smoke chipyard-scala synth-rtl memory-models reproduce audit clean

bootstrap:
	./scripts/bootstrap.sh

test:
	$(PYTHON) -m pytest

test-ml:
	.venv-ml/bin/python -m pytest -q tests_ml

lint-rtl:
	./scripts/lint_rtl.sh

rtl-smoke:
	./scripts/run_rtl_smoke.sh

chipyard-smoke:
	./scripts/check_chipyard_overlay.sh

chipyard-scala:
	$(PYTHON) scripts/check_chipyard_scala.py

synth-rtl:
	./scripts/bootstrap_rtl_tools.sh
	$(PYTHON) scripts/synthesize_rtl.py

memory-models:
	./scripts/bootstrap_memory_tools.sh
	$(PYTHON) scripts/run_memory_models.py
	$(PYTHON) scripts/run_drampower.py

reproduce:
	$(PYTHON) scripts/reproduce.py --output artifacts/results

audit:
	$(PYTHON) scripts/audit_quality.py --quality-dir artifacts/results/quality
	$(PYTHON) scripts/audit_reproduction.py --results artifacts/results

clean:
	rm -rf build rtl/obj_dir artifacts/generated
