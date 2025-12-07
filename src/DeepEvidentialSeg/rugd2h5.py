import h5py
import numpy as np
import os
import tyro

def create_rugd_h5(root_dir: str, save_path: str):
    """
    Create an HDF5 file containing the paths to all RUGD images in the specified directory.

    Args:
        root_dir (str): The root directory of RUGD dataset.
        save_path (str): The path where the HDF5 file will be saved.
    """
    rugd_files_path = []
    
    labels_dir = os.path.join(root_dir, "RUGD_annotations")
    images_dir = os.path.join(root_dir, "RUGD_frames-with-annotations")

    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        raise FileNotFoundError("Required directories do not exist in the specified root directory.")

    labels_subdir = set(os.listdir(labels_dir))
    images_subdir = set(os.listdir(images_dir))

    available_subdir = labels_subdir.intersection(images_subdir)

    for subdir in available_subdir:
        labels_subdir_path = os.path.join(labels_dir, subdir)
        images_subdir_path = os.path.join(images_dir, subdir)

        sub_labels = sorted([ file_path for file_path in os.listdir(labels_subdir_path) if file_path.endswith('.png') ])
        sub_images = sorted([ file_path for file_path in os.listdir(images_subdir_path) if file_path.endswith('.png') ])

        if len(sub_labels) != len(sub_images):
            raise ValueError(f"Mismatch in number of label and image files in subdirectory '{subdir}'.")
        
        for label_file, image_file in zip(sub_labels, sub_images):
            label_path = os.path.join(labels_subdir_path, label_file)
            image_path = os.path.join(images_subdir_path, image_file)

            rugd_files_path.append((image_path, label_path))

    
    data_len = len(rugd_files_path)
    with h5py.File(save_path, 'w') as f:
        dt_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
        f.create_dataset('images', (data_len, ), dtype=dt_uint8)
        f.create_dataset('labels', (data_len, ), dtype=dt_uint8)

        for i, (image_path, label_path) in enumerate(rugd_files_path):
            with open(image_path, 'rb') as img_file:
                f['images'][i] = np.frombuffer(img_file.read(), dtype=np.uint8)
            with open(label_path, 'rb') as lbl_file:
                f['labels'][i] = np.frombuffer(lbl_file.read(), dtype=np.uint8)
            
            # Logging progress every 100 files
            if i % 100 == 0:
                print(f"Processed {i}/{data_len} files...")

def main():
    tyro.cli(create_rugd_h5)

if __name__ == "__main__":
    main()

