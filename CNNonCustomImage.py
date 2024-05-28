import os
import pandas as pd
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping

data_dir = "CustomImage\cell_images"
# print(os.listdir(data_dir))

test_path = r"CustomImage\cell_images\test"
train_path = r"CustomImage\cell_images\train"

# print(os.listdir(test_path))
# print(os.listdir(train_path))

# print(os.listdir(train_path+'\parasitized')[0])

para_cell = train_path + "\parasitized" + "\C100P61ThinF_IMG_20150918_144104_cell_162.png"

# print(imread(para_cell))
# plt.imshow(imread(para_cell))
# plt.show()

# print(os.listdir(train_path+r'\uninfected')[0])

uninfected_cell = train_path + r"\uninfected" + r"\C100P61ThinF_IMG_20150918_144104_cell_128.png"
# print(imread(uninfected_cell))
# plt.imshow(imread(uninfected_cell))
# plt.show()

# print(len(os.listdir(train_path+'\parasitized')))
# print(len(os.listdir(train_path+r'\uninfected')))
# print(len(os.listdir(test_path+'\parasitized')))
# print(len(os.listdir(test_path+r'\uninfected')))

# print(help(ImageDataGenerator))

image_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
# plt.imshow(imread(para_cell))
# plt.show()
# plt.imshow(image_gen.random_transform(imread(para_cell)))
# plt.show()

image_shape = (130, 130, 3)
print(image_gen.flow_from_directory(train_path))

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', mode=min, patience=2)
batch_size = 16

# print(image_shape)
# print(image_shape[:2])
train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2], color_mode='rgb', batch_size=batch_size, class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,target_size=image_shape[:2], color_mode='rgb', batch_size=batch_size, class_mode='binary', shuffle=False)
# print(train_image_gen.class_indices)

# history = model.fit_generator(train_image_gen, epochs=20, validation_data=test_image_gen, callbacks=[early_stop])

# model.save('CNNonCustomimage.h5')
# losses = pd.DataFrame(history.history)
# losses.to_csv('CNNonCustomimage.csv')

later_model = load_model('CustomImage\malaria_detector.h5')
