from imports import *
from data_vis import train_data, class_names, validation_data, train_path, val_path, writer, train_writer, test_writer, test_data
from data_logging import create_callback
from disp_training import plot_loss_curves
from confusion_matrix import plot_confusion_matrix


IMG_SIZE = (400, 400)

# Defining the base model
efficientNet_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top = False,
                                                     weights = 'imagenet',
                                                     input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3))

efficientNet_model.summary()

efficientNet_model.trainable = False

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                                         tf.keras.layers.RandomRotation(0.2),
                                         tf.keras.layers.RandomZoom(0.2),
                                         tf.keras.layers.RandomHeight(0.2),
                                         tf.keras.layers.RandomWidth(0.2),
                                         ],
                                         name = "data_augmentation")


for image, _ in train_data.take(1):  
    plt.figure(figsize=(5, 5))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')

plt.show()


name = "EfficientNetB7"

efficientNet_model = tf.keras.Sequential([tf.keras.Input(shape=(None, None, 3), name="input_layer"),
                                      data_augmentation,
                                      # Switch this variable with the model you would like to train.
                                      efficientNet_model,
                                      tf.keras.layers.GlobalAveragePooling2D(),
                                      tf.keras.layers.Dense(128, activation = "relu"),
                                      tf.keras.layers.Dropout(0.2),
                                      tf.keras.layers.Dense(len(class_names), activation = "softmax"),
                                      ],
                                      name = name)

loss_fn = tf.keras.optimizers.Adam(learning_rate = 0.001)


efficientNet_model.compile(loss='categorical_crossentropy',
                              optimizer= loss_fn,
                              metrics=['accuracy'])

efficientNet_model.summary()

model_callback = create_callback(name)

EPOCH = 20

start_time = time.time()

# Verify dataset paths
print("Training Data Path:", train_path)
print("Validation Data Path:", val_path)

# Inspect data loading code
# Ensure that train_data and validation_data are correctly initialized
# and are not empty
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(validation_data))

# Check dataset shape
for images, labels in train_data.take(1):
    print("Training batch shape:", images.shape)
    print("Training batch labels shape:", labels.shape)

for images, labels in validation_data.take(1):
    print("Validation batch shape:", images.shape)
    print("Validation batch labels shape:", labels.shape)

efficientnet_model_history = efficientNet_model.fit( train_data,
                    epochs = EPOCH,
                    steps_per_epoch=len(train_data),
                    validation_data=validation_data, 
                    validation_steps=len(validation_data),
                    callbacks = model_callback)

# Saving the trained model
efficientNet_model.save("./EfficientNet_Model.keras")

writer.close()
train_writer.close()
test_writer.close()

print(efficientnet_model_history.history.keys())


# Plotting the training of the model
plot_loss_curves(efficientnet_model_history)

test_loss, test_accuracy = efficientNet_model.evaluate(test_data, verbose = 0)

print("Test loss: {:.5f}".format(test_loss))
print("Test accuracy: {:.2f}%".format((test_accuracy * 100)))
      
# Classification metrics
pred_probs = efficientNet_model.predict(test_data, verbose=1)

pred_classes = pred_probs.argmax(axis=1)
pred_classes[:10]

y_labels = []

for images, labels in test_data.unbatch():
    y_labels.append(labels.numpy().argmax())

y_labels[:10]

print('Classification Report \n')
target_names = class_names
print(classification_report(y_labels, pred_classes, target_names=target_names))

cm = confusion_matrix(y_labels, pred_classes)
plot_confusion_matrix(cm, class_names)