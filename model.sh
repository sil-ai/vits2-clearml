# Execute preprocess_clearml.py
echo "Running preprocessing script..."
python3 preprocess_clearml.py
# Execute the training script
echo "Running training script..."
python3 train.py

echo "All scripts executed successfully."