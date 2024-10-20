# AI_fruit_QA
AI for optically detected fruit quality

# Install requirement packages
```
pip install -r requirements.txt
```
# Create 5 fold data for training
Ensure your data is in the correct format and location. You may need to preprocess the data or split it into training and validation sets.
```
python preprocess_data/K_fold.py
```

# Start training with pytorch lightning
```
python train --task brix --fold [1,2,3,4,5] -lr 0.002 -bs 64
```

# Make prediction by ensemble 5 fold weights
```
python predict --task brix -bs 64
```
