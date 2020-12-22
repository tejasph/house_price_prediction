# house price prediction pipeline
# author: Tejas and Neel Phaterpekar

all: data/cleaned_train.csv data/X_train.csv data/X_valid.csv data/y_train.csv data/y_valid.csv

data/cleaned_train.csv: src/fill_missing.py
	python src/fill_missing.py --train_path=data/train.csv

data/X_train.csv data/X_valid.csv data/y_train.csv data/y_valid.csv: src/split_data.py
	python src/split_data.py --clean_train_path=data/cleaned_train.csv

clean:
	rm -rf data/cleaned_train.csv
	rm -rf data/X_train.csv
	rm -rf data/X_valid.csv
	rm -rf data/y_train.csv
	rm -rf data/y_valid.csv