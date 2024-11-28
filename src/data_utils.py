import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from src.config import (
    DATA_DIR,
    IMAGES_DIR,
    LABELS_FILE,
    SPLITS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)

def load_and_verify_data():
    # Load labels
    df = pd.read_csv(LABELS_FILE)
    
    required_columns = {'Image', 'Class'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Excel file is missing the following required columns: {missing}")
    
    # Verify images
    image_filenames = set(os.listdir(IMAGES_DIR))
    excel_filenames = set(df['Image'])
    
    # Check for missing images
    missing_images = excel_filenames - image_filenames
    if missing_images:
        print("Warning: Missing images listed in Excel:")
        for img in missing_images:
            print(f" - {img}")
        df = df[df['Image'].isin(image_filenames)]
        print(f"Filtered DataFrame to {len(df)} labeled images.")
    # else:
    #     print("All labeled images are present in the Sliced_Images directory.")
    
    # Summary
    unlabeled_images = image_filenames - excel_filenames
    # print(f"Total labeled images: {len(df)}")
    # print(f"Total images: {len(image_filenames)}")
    # print(f"Total unlabeled images: {len(unlabeled_images)}")
    # print(f"Found {df['Class'].nunique()} unique classes: {sorted(df['Class'].unique())}")
    
    return df, image_filenames, unlabeled_images

def split_data(df):
    # Define split ratios
    train_ratio = TRAIN_RATIO
    val_ratio = VAL_RATIO
    test_ratio = TEST_RATIO
    
    # Ensure ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Split ratios must sum to 1."
    
    # Features and labels
    X = df['Image']
    y = df['Class']
    
    # First split: Train and Temp (Val + Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=42)
    train_idx, temp_idx = next(sss1.split(X, y))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
    
    # Second split: Validation and Test
    val_size = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
    
    # Print split sizes
    # print(f"Training set: {len(X_train)} images")
    # print(f"Validation set: {len(X_val)} images")
    # print(f"Test set: {len(X_test)} images")
    
    return X_train, X_val, X_test

def save_split(file_list, filename):
    filepath = os.path.join(SPLITS_DIR, filename)
    with open(filepath, 'w') as f:
        for item in file_list:
            f.write(f"{item}\n")
    # print(f"Saved {len(file_list)} entries to {filepath}")

def create_and_save_splits(df, image_filenames, unlabeled_images):
    # Create splits directory
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    # Split the data
    X_train, X_val, X_test = split_data(df)
    
    # Save labeled splits
    save_split(X_train.tolist(), 'train_labeled.txt')
    save_split(X_val.tolist(), 'val_labeled.txt')
    save_split(X_test.tolist(), 'test.txt')
    
    # Prepare pre-training split: Unlabeled + Train Labeled + Val Labeled
    pretrain_filenames = unlabeled_images.union(set(X_train.tolist())).union(set(X_val.tolist()))
    save_split(pretrain_filenames, 'train_unlabeled.txt')

if __name__ == "__main__":
    df, image_filenames, unlabeled_images = load_and_verify_data()
    create_and_save_splits(df, image_filenames, unlabeled_images)
