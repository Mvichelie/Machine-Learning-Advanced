import os
import random
import argparse

# a function to collect files from the dataset directory with language, resolution, and style filtering
def collect_files(data_dir, language, resolution_filter=None, style_filter=None):
    files = []

    language_dirs = {
        'Thai': os.path.join(data_dir, 'Thai'),
        'English': os.path.join(data_dir, 'English')
    }

    selected_languages = []
    if language.lower() == 'thai':
        selected_languages.append('Thai')
    elif language.lower() == 'english':
        selected_languages.append('English')
    elif language.lower() == 'both':
        selected_languages.extend(['Thai', 'English'])
    else:
        print(f"Language not included please choose between Thai, English or both: {language}")
        return files

    for lang in selected_languages:
        lang_dir = language_dirs.get(lang)
        if not lang_dir or not os.path.exists(lang_dir):
            print(f"The language directory {lang_dir} does not exist.")
            continue

        # going over character folders
        for character_folder in os.listdir(lang_dir):
            character_path = os.path.join(lang_dir, character_folder)
            if not os.path.isdir(character_path):
                continue

            # going over resolution folders (200, 300, 400)
            for res_folder in os.listdir(character_path):
                res_path = os.path.join(character_path, res_folder)
                if not os.path.isdir(res_path):
                    continue

                # applying resolution filter
                if resolution_filter and res_folder not in resolution_filter:
                    continue

                # going over style folders ( bold, italic, normal..)
                for style_folder in os.listdir(res_path):
                    style_path = os.path.join(res_path, style_folder)
                    if not os.path.isdir(style_path):
                        continue

                    # style filter
                    if style_filter and style_folder not in style_filter:
                        continue

                    # going over image files
                    for filename in os.listdir(style_path):
                        if filename.endswith('.bmp'):
                            file_path = os.path.join(style_path, filename)
                            files.append((file_path, res_folder, style_folder))

    return files

#a function to split dataset into training, validation, and test sets
def split_dataset(files, train_ratio=0.7, val_ratio=0.2):
    #shuffle the dataset for randomness
    random.shuffle(files)  
    total_files = len(files)
    train_count = int(train_ratio * total_files)
    val_count = int(val_ratio * total_files)


    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    return train_files, val_files, test_files

def save_file_list(file_list, file_name):
    with open(file_name, 'w') as f:
        for file_path, resolution, style in file_list:
            f.write(f"{file_path}, {resolution}, {style}\n")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into training, validation, and test sets.")
    parser.add_argument('--data_dir', type=str, required=True, help="The directory where the OCR dataset is located.")
    parser.add_argument('--output_dir', type=str, default='./output', help="The directory where the train, val, and test file lists are saved.")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Ratio of data used for training.")
    parser.add_argument('--val_ratio', type=float, default=0.2, help="Ratio of data used for validation.")
    parser.add_argument('--language', type=str, default='Thai', choices=['Thai', 'English', 'both'], help="Language to use for OCR: 'Thai', 'English', or 'both'.")
    parser.add_argument('--resolutions', nargs='*', default=None, help="Filter by resolution (e.g., 200 300 400). Leave empty for all resolutions.")
    parser.add_argument('--styles', nargs='*', default=None, help="Filter by style (e.g., normal bold italic bold_italic). Leave empty for all styles.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # collecting and splitting the dataset
    files = collect_files(args.data_dir, args.language, resolution_filter=args.resolutions, style_filter=args.styles)
    if not files:
        print("Oops, no image files found.")
        return

    train_files, val_files, test_files = split_dataset(files, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # Save the split datasets to files
    save_file_list(train_files, os.path.join(args.output_dir, 'train_files.txt'))
    save_file_list(val_files, os.path.join(args.output_dir, 'val_files.txt'))
    save_file_list(test_files, os.path.join(args.output_dir, 'test_files.txt'))

    print(f"The number of training samples: {len(train_files)}")
    print(f" The number of validation samples: {len(val_files)}")
    print(f"The number of test samples: {len(test_files)}")

if __name__ == "__main__":
    main()
