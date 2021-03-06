# house price prediction pipeline
# author: Tejas and Neel Phaterpekar

all: data/cleaned_train.csv \
data/X_train.csv \
data/X_valid.csv \
data/y_train.csv \
data/y_valid.csv \
data/X_train_scaled.csv \
models/dummy_reg.pkl \
models/base*.pkl \
models/opt_rf.pkl \
models/opt_svr.pkl \
models/opt_gbr.pkl \
results/model_table.csv \
data/cleaned_test.csv \
data/X_test_scaled.csv

# Deal with missing data
data/cleaned_train.csv: src/fill_missing.py
	python src/fill_missing.py --train_path=data/train.csv --write_name=train

# Splits the data
data/X_train.csv data/X_valid.csv data/y_train.csv data/y_valid.csv: src/split_data.py data/cleaned_train.csv
	python src/split_data.py --clean_train_path=data/cleaned_train.csv

# Preprocessing Step
data/X_train_scaled.csv data/X_valid_scaled.csv: src/preprocessor.py data/X_train.csv data/X_valid.csv
	python src/preprocessor.py

# Train Dummy Model (need to change to scaled data)
models/dummy_reg.pkl: data/X_train_scaled.csv data/X_valid_scaled.csv data/y_train.csv data/y_valid.csv
	python src/baseline_score.py

# Train Baseline Models
models/base*.pkl: src/baseline_models.py data/X_train_scaled.csv data/y_train.csv 
	python src/baseline_models.py

# Optimize RF (can potentially remove X_valid, yvalid)
models/opt_rf.pkl: src/optimize_rf.py data/X_train_scaled.csv data/X_valid_scaled.csv data/y_train.csv data/y_valid.csv
	python src/optimize_rf.py

# Optimize SVR
models/opt_svr.pkl: src/optimize_svr.py data/X_train_scaled.csv data/y_train.csv
	python src/optimize_svr.py

# Optimize GradientBoostingRegressor
models/opt_gbr.pkl: src/optimize_gbr.py data/X_train_scaled.csv data/y_train.csv
	python src/optimize_gbr.py

# Create Results Table for models
results/model_table.csv: src/model_table_init.py models/dummy_reg.pkl
	python src/model_table_init.py

##################################################################################################################
# Prepare Test data
data/cleaned_test.csv: src/fill_missing.py
	python src/fill_missing.py --train_path=data/test.csv --write_name=test

# Process data
data/X_test_scaled.csv: src/preprocess_test.py data/cleaned_test.csv data/cleaned_train.csv
	python src/preprocess_test.py

# Remove all files
clean:
	rm -rf data/cleaned_train.csv
	rm -rf data/X_train.csv
	rm -rf data/X_valid.csv
	rm -rf data/y_train.csv
	rm -rf data/y_valid.csv
	rm -rf models/*.pkl
	rm -rf results/model_table.csv
	rm -rf data/cleaned_test.csv
	rm -rf data/X_test_scaled.csv



