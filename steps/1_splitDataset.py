import os
import yaml
import random
import shutil

list_path = ['datasets', 
             'datasets/train', 'datasets/train/images', 'datasets/train/labels',
             'datasets/validation', 'datasets/validation/images', 'datasets/validation/labels',
             'datasets/test', 'datasets/test/images', 'datasets/test/labels']

for path in list_path:
    if not os.path.exists(path):
        os.mkdir(path)

params = yaml.safe_load(open("params.yaml"))["splitDataset"]
train_proportion = params['train_proportion']
validation_proportion = params['validation_proportion']
test_proportion = params['test_proportion']
division = params['division']
seed = params['seed']

random.seed(seed)


# If the division is false split the dataset by yourself, otherwise is random according to the proportion you have set up
if division == 'false':
    pass

else:
    path_dataset_images = 'data/images'
    path_dataset_labels = 'data/fixed_annotations'
    path_train_images = 'datasets/train/images'
    path_train_labels = 'datasets/train/labels'
    path_validation_images = 'datasets/validation/images'
    path_validation_labels = 'datasets/validation/labels'
    path_test_images = 'datasets/test/images'
    path_test_labels = 'datasets/test/labels'

    list_images = os.listdir(path_dataset_images)


    # VALIDATION
    n_images_validation = round(validation_proportion * len(os.listdir(path_dataset_images)), )
    list_img_validation = random.sample(list_images, n_images_validation)

    for image in list_img_validation:
        name_file_label = image.split('.')[0] + '.txt'
        shutil.copy(f'{path_dataset_images}/{image}', f'{path_validation_images}/{image}')
        shutil.copy(f'{path_dataset_labels}/{name_file_label}', f'{path_validation_labels}/{name_file_label}')


    # TEST
    list_images_remained = list(set(list_images) - set(list_img_validation))
    n_images_test = round(test_proportion * len(os.listdir(path_dataset_images)), )
    list_img_test = random.sample(list_images_remained, n_images_test)

    for image in list_img_test:
        name_file_label = image.split('.')[0] + '.txt'
        shutil.copy(f'{path_dataset_images}/{image}', f'{path_test_images}/{image}')
        shutil.copy(f'{path_dataset_labels}/{name_file_label}', f'{path_test_labels}/{name_file_label}')


    # TRAINING
    list_images_remained1 = list(set(list_images_remained) - set(list_img_test))

    for image in list_images_remained1:
        name_file_label = image.split('.')[0] + '.txt'
        shutil.copy(f'{path_dataset_images}/{image}', f'{path_train_images}/{image}')
        shutil.copy(f'{path_dataset_labels}/{name_file_label}', f'{path_train_labels}/{name_file_label}')

