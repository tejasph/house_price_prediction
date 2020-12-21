# house price prediction pipeline
# author: Tejas and Neel Phaterpekar

all: data/cleaned_train.csv

data/cleaned_train.csv: src/fill_missing.py
	python src/fill_missing.py --train_path=data/train.csv

clean:
	rm -rf data/cleaned_train.csv