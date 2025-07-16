import os

# Paths to your folders containing the images and XML files
image_folder = 'D:/labelImg_Rename_Test'
xml_folder = 'D:/labelImg_Rename_Test'

def rename_images_and_xmls(image_folder, xml_folder, prefix, class_name, start_number=1):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".JPG") or f.endswith(".jpg")]
    
    counter = start_number
    for image_filename in image_files:
        # Get the base name without the extension
        base_name = os.path.splitext(image_filename)[0]
        
        # Define the new name for the image and XML
        new_name = f"{prefix}_{class_name}_{counter}"
        new_image_name = f"{new_name}{os.path.splitext(image_filename)[1]}"  # Keep original extension (JPG/PNG)
        new_xml_name = f"{new_name}.xml"
        
        # Full paths for image and XML
        old_image_path = os.path.join(image_folder, image_filename)
        old_xml_path = os.path.join(xml_folder, f"{base_name}.xml")
        new_image_path = os.path.join(image_folder, new_image_name)
        new_xml_path = os.path.join(xml_folder, new_xml_name)
        
        # Rename the image
        os.rename(old_image_path, new_image_path)
        print(f"Renamed image {image_filename} to {new_image_name}")
        
        # Rename the corresponding XML if it exists
        if os.path.exists(old_xml_path):
            os.rename(old_xml_path, new_xml_path)
            print(f"Renamed XML {base_name}.xml to {new_xml_name}")
        else:
            print(f"No matching XML found for {image_filename}")

        counter += 1

# Call the function for your images and XMLs
rename_images_and_xmls(image_folder, xml_folder, 'Rotten', 'Pear', 1)
