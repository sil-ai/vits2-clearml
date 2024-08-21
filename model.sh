echo "**********************************"
echo "Executing Vits 2 pipeline"
echo "**********************************"
echo "----------------------------------"
echo "Uploading data to clearml..."
# Upload data to clearml
python3 upload_step.py
echo "Data uploaded to clearml."
echo "----------------------------------"
# Execute preprocess_clearml.py
echo "Running preprocessing script..."
python3 preprocess_step.py
echo "Preprocessing script executed successfully."
echo "----------------------------------"
# Execute the training script
echo "Running training script..."
python3 train.py
echo "Training script executed successfully."
echo "----------------------------------"

echo "----------- FINISHED --------------"

echo "All scripts executed successfully."