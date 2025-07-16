import os
import glob
import xml.etree.ElementTree as ET

def count_labels(xml_folder, class1, class2):
    class1_count = 0
    class2_count = 0

    for xml_file in glob.glob(os.path.join(xml_folder, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            class_name = member.find('name').text
            if class_name == class1:
                class1_count += 1
            elif class_name == class2:
                class2_count += 1

    return class1_count, class2_count

# Paths to your folders containing the XML files
xml_folder = 'C:/Users/roger/OneDrive/Documents/Tensorflow/workspace/training_demo_7/images/manual_rebalance'

# Class names
class1 = 'jeruksemirotten'
class2 = 'pir-semirotten'

# Count the number of labels for each class
class1_count, class2_count = count_labels(xml_folder, class1, class2)

print(f"Number of labels for {class1}: {class1_count}")
print(f"Number of labels for {class2}: {class2_count}")