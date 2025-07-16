import os
import glob
import shutil
import xml.etree.ElementTree as ET

def count_labels(xml_folder, class_name):
    count = 0
    for xml_file in glob.glob(os.path.join(xml_folder, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member.find('name').text == class_name:
                count += 1
    return count

def select_balanced_images(xml_folder, image_folder, class1, class2, output_folder, num_labels):
    selected_images = set()
    class1_count = 0
    class2_count = 0

    for xml_file in glob.glob(os.path.join(xml_folder, '*.xml')):
        if class1_count >= num_labels and class2_count >= num_labels:
            break

        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_filename = root.find('filename').text
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            continue

        class1_labels = sum(1 for member in root.findall('object') if member.find('name').text == class1)
        class2_labels = sum(1 for member in root.findall('object') if member.find('name').text == class2)

        # Ensure images with both classes are included, but do not exceed the target number of labels
        if class1_labels > 0 and class2_labels > 0:
            if class1_count + class1_labels <= num_labels and class2_count + class2_labels <= num_labels:
                selected_images.add(image_filename)
                class1_count += class1_labels
                class2_count += class2_labels
        else:
            if class1_count < num_labels and class1_labels > 0:
                selected_images.add(image_filename)
                class1_count += class1_labels

            if class2_count < num_labels and class2_labels > 0:
                selected_images.add(image_filename)
                class2_count += class2_labels

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_filename in selected_images:
        shutil.copy(os.path.join(image_folder, image_filename), os.path.join(output_folder, image_filename))
        shutil.copy(os.path.join(xml_folder, os.path.splitext(image_filename)[0] + '.xml'), os.path.join(output_folder, os.path.splitext(image_filename)[0] + '.xml'))

    print(f"Selected {len(selected_images)} images with {num_labels} labels for each class.")

# Paths to your folders containing the images and XML files
xml_folder = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_7/images/unsplit'
image_folder = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_7/images/unsplit'
output_folder = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_7/images/rebalanced'

# Class names
class1 = 'jeruksemirotten'
class2 = 'pir-semirotten'

# Count the number of labels for each class
class1_count = count_labels(xml_folder, class1)
class2_count = count_labels(xml_folder, class2)

print(f"Number of labels for {class1}: {class1_count}")
print(f"Number of labels for {class2}: {class2_count}")

# Determine the number of labels to balance the dataset
num_labels = min(class1_count, class2_count)

# Select balanced images
select_balanced_images(xml_folder, image_folder, class1, class2, output_folder, num_labels)

# Verify the number of labels in the output folder
output_class1_count = count_labels(output_folder, class1)
output_class2_count = count_labels(output_folder, class2)

print(f"Number of labels for {class1} in the output folder: {output_class1_count}")
print(f"Number of labels for {class2} in the output folder: {output_class2_count}")