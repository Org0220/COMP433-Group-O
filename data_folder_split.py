import os
import pandas as pd

# Creating file paths for data to be read
data_dir =  os.getcwd()
images_dir = os.path.join(data_dir, 'Sliced_Images')
labels_file = os.path.join(data_dir, 'image_class_mapping.xlsx')


# Loading excel labels with pandas and checking if classes are read properly
df = pd.read_excel(labels_file)

required_columns = {'Image', 'Class'}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(f"Excel file is missing the following required columns: {missing}")

# Check how many total images and total labels
image_filenames = set(os.listdir(images_dir))
excel_filenames = set(df['Image'])
print(len(excel_filenames))
print(len(image_filenames))

# Check if all excel labeled data correspond to an image
missing_images = excel_filenames - image_filenames
if missing_images:
    print("Warning: The following images listed in the Excel file are missing in the Sliced_Images directory:")
    for img in missing_images:
        print(f" - {img}")
    df = df[df['Image'].isin(image_filenames)]
    print(f"Filtered DataFrame to {len(df)} labeled images.")
else:
    print("All labeled images are present in the Sliced_Images directory.")

# Print total number of unlabeled data
unlabeled_images = image_filenames - excel_filenames
print(f"Total unlabeled images: {len(unlabeled_images)}")

# Print total number of unique classes
classes = sorted(df['Class'].unique())
print(f"Found {len(classes)} unique classes: {classes}")

