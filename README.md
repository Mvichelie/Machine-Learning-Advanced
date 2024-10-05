# Machine-Learning-Advanced Assignment 1 
To split the data set, I have a script called split_dataset.py. 
run this in the terminal:
python split_dataset.py --data_dir /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet --output_dir ./output --language Thai --train_ratio 0.7 --val_ratio 0.2

another example commnand:
python split_dataset.py --data_dir /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet --output_dir ./output --language Thai --train_ratio 0.7 --val_ratio 0.2 --resolutions 200 300 400 --styles normal bold italic

Args in the split_dataset.py script:

--data_dir: Specifies the path to the original OCR dataset. the directory /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet, where the dataset can be accessed. 

--output_dir: The directory where the resulting file lists (.txt) for training, validation,  and testing are saved.

--language: The language(s) to include in the OCR task which are Thai, English, or both. 

--resolutions: (Optional) To filter the dataset by image resolution (e.g., 200, 300, or 400 DPI). You can specify yourself. 

--styles: (Optional) Filter the dataset by font style (e.g., normal, bold, italic, or bold_italic). Multiple styles can be specified.

--train_ratio: Proportion of the dataset to be used for training. (0.7)

--val_ratio: Proportion of the dataset to be used for validation. (0.2) 


