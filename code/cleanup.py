from imports import *

print("hello world")

data = './Data/'
experiment_name = 'Transfer Learning'
tr_data = './UsableData/train'
te_data = './UsableData/test'
va_data = './UsableData/val'

print(os.listdir(data))

# prepare_data_emission_tracker = EmissionsTracker()
# prepare_data_emission_tracker.start()


# Things to read into about this function:
#   os.unlink

# def remove_folder_contents(folder):

#     for file in os.listdir(folder):
#         file_path = os.path.join(folder, file)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 remove_folder_contents(file_path)
#                 os.rmdir(file_path)
#         except Exception as e:
#             print(e)


# model_data = './UsableData'
# remove_folder_contents(model_data)

# Removing incompatible data formats and corruted data
count = 0
image_extensions = ['.png', '.jpg']
img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]

for filepath in Path(tr_data).glob('*'):

    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)

        if img_type is None:
            print(f"{filepath} is name an image")

        if img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by tensorflow")
            os.remove(filepath)
            count += 1

print(f"Removed {count} images")

print("\nLooking for incompatble images")

for filepath in Path(va_data).glob('*'):
    for image_name in os.listdir(filepath):
        # Construct the full path to the image file
        image_path = os.path.join(filepath, image_name)
        try:
            # Decode the image using TensorFlow
            decoded_image = tf.io.read_file(image_path)
            decoded_image = tf.image.decode_image(decoded_image)
        except tf.errors.InvalidArgumentError as e:
            # If decoding fails, print the error message
            print("Error decoding image:", e)
            print("Image path:", image_path)


