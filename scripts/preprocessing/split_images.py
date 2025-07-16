import os
import random
import shutil
import math
import argparse

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_dataset(source_dir, dest_dir, train_ratio, valid_ratio, test_ratio):
    source_dir = source_dir.replace('\\', '/')
    dest_dir = dest_dir.replace('\\', '/')
    
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'valid')
    test_dir = os.path.join(dest_dir, 'test')

    create_dir_if_not_exists(train_dir)
    create_dir_if_not_exists(valid_dir)
    create_dir_if_not_exists(test_dir)

    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)
    print(f"Total number of images found: {num_images}")

    num_train_images = math.ceil(train_ratio * num_images)
    num_valid_images = math.ceil(valid_ratio * num_images)
    num_test_images = num_images - num_train_images - num_valid_images

    random.shuffle(images)

    train_images = images[:num_train_images]
    valid_images = images[num_train_images:num_train_images + num_valid_images]
    test_images = images[num_train_images + num_valid_images:]

    def copy_files(file_list, target_dir):
        for filename in file_list:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            if os.path.exists(os.path.join(source_dir, xml_filename)):
                shutil.copy(os.path.join(source_dir, xml_filename), os.path.join(target_dir, xml_filename))
            else:
                print(f"Warning: XML file for {filename} not found.")

    copy_files(train_images, train_dir)
    copy_files(valid_images, valid_dir)
    copy_files(test_images, test_dir)

    print(f"Dataset split completed: {len(train_images)} train, {len(valid_images)} valid, {len(test_images)} test images.")
    print(f"Total images processed: {len(train_images) + len(valid_images) + len(test_images)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, valid, and test sets.")
    parser.add_argument("source_dir", type=str, help="Path to the source directory containing images and XML files.")
    parser.add_argument("dest_dir", type=str, help="Path to the destination directory where the split folders will be created.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training set.")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="Ratio of validation set.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test set.")

    args = parser.parse_args()

    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-9):
        raise ValueError("The sum of train_ratio, valid_ratio, and test_ratio must be 1.0")

    split_dataset(args.source_dir, args.dest_dir, args.train_ratio, args.valid_ratio, args.test_ratio)