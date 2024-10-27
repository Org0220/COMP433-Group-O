import os
import pandas as pd
# def list_file_names(directory):
#     return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# # Example usage
# directory_path = "./Huron_data/Sliced_Images"
# file_names = list_file_names(directory_path)
# print(file_names)

def get_image_paths_from_excel(excel_path):
    """
    Reads an Excel file and extracts the image paths into a list.
    
    Args:
        excel_path (str): Path to the Excel file.

    Returns:
        List[str]: List of image paths.
    """
    try:
        # Load the Excel file
        data = pd.read_excel(excel_path)

        # Ensure the 'image_path' column exists
        if 'Image' not in data.columns:
            raise ValueError("The Excel file must contain an 'image_path' column.")

        # Extract image paths into a list
        image_paths = data['Image'].tolist()
        image_paths.map(lambda x: x.split('.')[0])
        return image_paths

    except Exception as e:
        print(f"Error: {e}")
        return []

# Example usage
excel_file = 'Huron_data/Labeled_Data.xlsx'  # Replace with your Excel file path
image_paths = get_image_paths_from_excel(excel_file)

print(f"Loaded {len(image_paths)} image paths:")
print(image_paths)