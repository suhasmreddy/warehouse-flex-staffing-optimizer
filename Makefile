.PHONY: install test download reproduce reproduce-cached verify-offline clean

install:
	python3 -m pip install -r requirements.txt

test:
	@if command -v pytest >/dev/null 2>&1; then \
		pytest -q; \
	else \
		python3 -m unittest discover -s tests -p "test_*.py" -v; \
	fi

download:
	python3 -m src.cli download --config configs/base.yml

# Full raw-data reproduction (downloads 43.5 MB online_retail_II.xlsx -> DuckDB SQL -> Ridge -> Pyomo/HiGHS)
reproduce:
	bash scripts/reproduce.sh

# Fast offline reproduction from the committed 525-day UCI daily_workload.csv spine (<3s, no download needed)
reproduce-cached:
	python3 -m src.cli run \
		--config configs/base.yml \
		--from-daily-csv results/uci/daily_workload.csv \
		--output-dir results/uci \
		--dataset-mode uci_online_retail_ii
	python3 scripts/render_uci_report.py

verify-offline:
	python3 scripts/generate_synthetic_fixture.py
	python3 -m src.cli run \
		--config configs/base.yml \
		--excel-path data/fixtures/synthetic_retail_two_sheet_fixture.xlsx \
		--output-dir results/verification_smoke \
		--dataset-mode synthetic_verification_fixture \
		--allow-offline-fallback

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache results/verification_smoke
