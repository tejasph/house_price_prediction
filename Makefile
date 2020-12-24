# house price prediction pipeline
# author: Tejas and Neel Phaterpekar

all: data/cleaned_train.csv \
data/X_train.csv \
data/X_valid.csv \
data/y_train.csv \
data/y_valid.csv \
models/dummy_reg.pkl \
results/model_table.csv


# Deal with missing data
data/cleaned_train.csv: src/fill_missing.py
	python src/fill_missing.py --train_path=data/train.csv

# Preprocessing Step

# Splits the data
data/X_train.csv data/X_valid.csv data/y_train.csv data/y_valid.csv: src/split_data.py
	python src/split_data.py --clean_train_path=data/cleaned_train.csv


# Train Models
models/dummy_reg.pkl: data/X_train.csv data/X_valid.csv data/y_train.csv data/y_valid.csv
	python src/baseline_score.py

# Create Results Table for models
results/model_table.csv: src/model_table_init.py
	python src/model_table_init.py

clean:
	rm -rf data/cleaned_train.csv
	rm -rf data/X_train.csv
	rm -rf data/X_valid.csv
	rm -rf data/y_train.csv
	rm -rf data/y_valid.csv
	rm -rf results/model_table.csv
	rm -rf models/dummy_reg.pkl