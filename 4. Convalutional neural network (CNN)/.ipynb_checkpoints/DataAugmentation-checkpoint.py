#!/usr/bin/env python
# coding: utf-8

# ## For singal image

# In[2]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=35,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

img = load_img('C:\\Users\\Sorcim\\tensorflow deep learning\\4. Convalutional neural network (CNN)\\img\\image.jfif')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)


i = 0
for batch in datagen.flow(x, batch_size=1,
                         save_to_dir='C:\\Users\\Sorcim\\tensorflow deep learning\\4. Convalutional neural network (CNN)\\augmented\\', save_prefix='cat', save_format='jpg'):
    i += 1
    if i > 1:
        break


# ## For images in directory

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


for i in range(127):
    chosen_image = os.listdir('C:\\Users\\Sorcim\\Documents\\dataset\\cards_resized\\')[i]
    image_path ='C:\\Users\\Sorcim\\Documents\\dataset\\cards_resized\\' + chosen_image
    
    img = load_img(image_path) # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    
    i = 0
    
    
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='cards_resized', save_prefix='change', save_format='jpeg'):
        i += 1
        if i > 1:
             break

