

# Training

## Please perform the following in order.

1. Download dataset from https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

2. Within the downloaded dataset, move the "BraTS2020_TrainingData" folder into the "Dataset" folder, we ignore the ValidationData folder as it does not contain the mask files as the dataset was meant for a competition. Next, ensure that the "BraTS2020_TrainingData" folder contains the "MICCAI_BraTS2020_TrainingData" folder. This then contains all 369 data points, each of which consists of the 4 MRI modes and mask file. 

3. Download the saved models from the https://drive.google.com/file/d/1g80HLTz_TKZcVFGG5Tl8YfXDDn8Ploh2/view?usp=sharing

4. Move the models folder into the project folder, ensuring the folder is named "saved_models"

5. Run code using the below steps

## Run in Terminal

### Create virtual env
```
python3.9 -m venv venv 
```
### Activate Virtual Env
(mac)
```
source venv/bin/activate
```
 (windows command prompt)
```
venv\Scripts\activate
```
### Install required packages
```
pip install -r requirements.txt
```

## Run training.ipynb Jupyter Notebook
1. Ensure that the dataset_path variable is defined correctly. In both training.ipynb and comparison.ipynb 

2. Uncomment the third cell to generate the numpy dataset.

3. Run the proceeding cells sequentially. The notebook will visualise some data slices. 

4. The models as stated above are commented out in the training file. You may uncomment them to train the model yourself, or use the models in the downloaded "saved_models" folder. The cells after each model can be uncommented as well to display the loss against epochs graphs for each trained model, after its respective model has been trained.




# Application

1. Ensure that virtual environment is activated

2. Ensure the model_path variable is correct. In this case running the very last model saved as "./saved_models/multi_channel_further/model_5.pth"

3. Start the application with 
```
python3 app.py
```
4. Access any datapoint within the dataset folder, and upload the corresponding MRI mode to the correct box. For example the T1 mri scan should be uploaded using the button "T1".

5. Click the "Analyse" Button to use the model to create the predicted mask. 

6. Once completed, the mask can be observed. You may also choose to upload the actual mask (tagged as the seg.nii file) within data point folder. 

7. The sliders can be used to view the different layers of the model. 



# View Demo at 
https://www.youtube.com/watch?v=zRvSbnc