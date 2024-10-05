# Machine-Learning-Advanced Assignment 1 
Note, I have all of my scripts  and the output files in the server in my home as well the the ocd_model.pth. These are all located in a folder named scripts. I cannot push it here due to some access permissions. I have only run those scripts through the terminal and the server. 


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

For training the model/data, I have a file called train_model.py (You need to provide the training and validation file lists generated by the split_dataset.py script first then run the text below)
: python train_model.py --train_file ./output/train_files.txt --val_file ./output/val_files.txt --output_dir ./output --epochs 20 --batch_size 32 --learning_rate 0.001

Args in the train_model.py script:
--train_file: Path to the training file list generated by split_dataset.py.

--val_file: Path to the validation file list generated by split_dataset.py.

--output_dir: The directory where the trained model and logs will be saved.
--epochs: Number of epochs for training (default: 10).

--batch_size: Batch size for training (default: 32).

--learning_rate: Learning rate for the optimizer (default: 0.001).

--img_height: Height of the input images (default: 64).

--img_width: Width of the input images (default: 64).

For the testing/evaluation, I have a file called evaluate_model.py. This will evaluate the model on the validation dataset and provide metrics such as accuracy, precision, recall, and F1 score.
Run this in the terminal testing or evaluating the model across different styles and resolutions:
python evaluate_model.py --val_file ./output/val_files.txt --output_dir ./output --batch_size 32 --img_height 64 --img_width 64

#Notes when I ran it the first time this is the scores that I got.These metrics were calculated across the validation dataset based on all the different styles and resolutions present in val_files.txt.: 
Accuracy: 0.9867
Precision: 0.9868
Recall: 0.9867
F1 Score: 0.9867

##Experiments
if you want to experiment with
Lang=Thai Style: Normal Text, 200 DPI
Training Data: Thai normal text, 200 dpi
Testing Data: Thai normal text, 200 dpi
run this :

python split_dataset.py --data_dir /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet --output_dir ./output --language Thai --resolutions 200 --styles normal


python train_model.py --train_file ./output/train_files.txt --val_file ./output/val_files.txt --output_dir ./output --epochs 20 --batch_size 32 --learning_rate 0.001


python evaluate_model.py --val_file ./output/val_files.txt --output_dir ./output --batch_size 32 --img_height 64 --img_width 64

When I run this experiment this is what i got:
Accuracy: 0.9361
Precision: 0.9387
Recall: 0.9361
F1 Score: 0.9355

For both languages:
python split_dataset.py --data_dir /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet --output_dir ./output --language both --resolutions 200 300 400 --styles normal

python train_model.py --train_file ./output/train_files.txt --val_file ./output/val_files.txt --output_dir ./output --epochs 20 --batch_size 32 --learning_rate 0.001

python evaluate_model.py --val_file ./output/val_files.txt --output_dir ./output --batch_size 32 --img_height 64 --img_width 64


##Challenges
I had a lot of trouble with the OCR model dimensions both during trianing and evaluating. I had come accross several errors with wrong dimension sizes numerous times before I was able to proceed. I had to rely on the demos from the class and several other resources in order to make sure I know what is going on. 


