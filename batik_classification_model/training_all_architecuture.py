import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten ,Dropout
from keras.models import Sequential, Model
from keras import applications
from keras.optimizers import SGD
from keras.callbacks import TensorBoard


train_data = 'dataset/training'
validation_data = 'dataset/testing'
model_applications = 'VGG16'

img_width = 128
img_height = 128
train_samples = 4000
validation_samples = 100
epochs = 3
batch_size = 32
num_classes = 11

#----------------------------------------VGG 16
base_model = applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

top_model = Flatten(name="flatten")(base_model.output)
top_model = Dense(512,activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes,activation="softmax")(top_model)

final_model = Model(inputs=base_model.input,outputs=top_model)


print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
final_model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Image Augementation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

#Model Fitting
callbacks = TensorBoard(log_dir='./Graph/VGG16')
H = final_model.fit_generator(
	train_generator,
	steps_per_epoch=train_samples // batch_size,
	validation_data=validation_generator,
	validation_steps=validation_samples // batch_size,
	epochs=epochs,
    callbacks=[callbacks])

#Evalution

# serialize the model to disk
print("[INFO] serializing network...")
final_model.save("fixmodelVGG16.h5")
train_generator.reset()
validation_generator.reset()

#------------------------------------------------------Mobile Net
base_model = applications.mobilenet_v2.MobileNetV2(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

top_model = Flatten(name="flatten")(base_model.output)
top_model = Dense(512,activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes,activation="softmax")(top_model)

final_model = Model(inputs=base_model.input,outputs=top_model)


print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
final_model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


#Model Fitting
callbacks = TensorBoard(log_dir='./Graph/MobileNetV2')
H = final_model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    epochs=epochs,
    callbacks=[callbacks])

#Evalution

# serialize the model to disk
print("[INFO] serializing network...")
final_model.save("fixmodelMobileNetV2.h5")
train_generator.reset()
validation_generator.reset()

#------------------------------------------------------Inception3
base_model = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

top_model = Flatten(name="flatten")(base_model.output)
top_model = Dense(512,activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes,activation="softmax")(top_model)

final_model = Model(inputs=base_model.input,outputs=top_model)


print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
final_model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

#Model Fitting
callbacks = TensorBoard(log_dir='./Graph/InveptionV3')
H = final_model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    epochs=epochs,
    callbacks=[callbacks])

#Evalution

# serialize the model to disk
print("[INFO] serializing network...")
final_model.save("fixmodelInceptionV3.h5")
train_generator.reset()
validation_generator.reset()

#------------------------------------------------------Nasnet
base_model = applications.nasnet.NASNetMobile(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

top_model = Flatten(name="flatten")(base_model.output)
top_model = Dense(512,activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(num_classes,activation="softmax")(top_model)

final_model = Model(inputs=base_model.input,outputs=top_model)


print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
final_model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


#Model Fitting
callbacks = TensorBoard(log_dir='./Graph/Nasnet')
H = final_model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    epochs=epochs,
    callbacks=[callbacks])

#Evalution

# serialize the model to disk
print("[INFO] serializing network...")
final_model.save("fixmodelNasNet.h5")
train_generator.reset()
validation_generator.reset()
