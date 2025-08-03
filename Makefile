.PHONY: install run clean

install:
	uv pip install -r requirements.txt

run:
	python my_first_pipeline.py

clean:
	rm -rf data/*