import os
import xml.etree.ElementTree as ET

def convert_to_yolo_format(xml_content, img_width, img_height, label_mapping):
    root = ET.fromstring(xml_content)
    yolo_format_str = ""

    for obj in root.findall('.//object'):
        name = obj.find('name').text
        if name in label_mapping:
            class_label = label_mapping[name]
        else:
            class_label = -1
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center_normalized = (xmin + (xmax - xmin) / 2) / img_width
        y_center_normalized = (ymin + (ymax - ymin) / 2) / img_height
        width_normalized = (xmax - xmin) / img_width
        height_normalized = (ymax - ymin) / img_height

        x_center_normalized = min(1, max(0, x_center_normalized))
        y_center_normalized = min(1, max(0, y_center_normalized))
        width_normalized = min(1, max(0, width_normalized))
        height_normalized = min(1, max(0, height_normalized))

        yolo_format_str += f"{class_label} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n"

    return yolo_format_str


def process_label_files_in_directory(annotations_in_path, annotations_out_path, label_mapping):
    for file_name in os.listdir(annotations_in_path):
        if file_name.endswith(".xml"):
            file_path = os.path.join(annotations_in_path, file_name)
            with open(file_path, 'r') as f:
                xml_content = f.read()

            img_width = int(ET.fromstring(xml_content).find(".//width").text)
            img_height = int(ET.fromstring(xml_content).find(".//height").text)

            yolo_format_str = convert_to_yolo_format(xml_content, img_width, img_height, label_mapping)
            out_file_name = file_name.replace(".xml", ".txt")
            yolo_file_path = os.path.join(annotations_out_path, out_file_name)
            with open(yolo_file_path, 'w') as f:
                f.write(yolo_format_str)

