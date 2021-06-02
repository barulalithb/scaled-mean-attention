# -*- coding: utf-8 -*-
"""
Created on Sun May 30 23:22:10 2021

@author: Lalith Bharadwaj
"""

import keras 
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input,Flatten,Dense,BatchNormalization,concatenate
from tensorflow.keras.layers import Attention,Lambda,Multiply,Reshape, Conv2D, Dropout
from keras.models import Model
from keras.layers.experimental.preprocessing import Normalization,Resizing,RandomContrast,RandomFlip,RandomTranslation,RandomRotation,RandomHeight,RandomWidth



def mean_2(x,alpha=2):
  return keras.backend.mean(x,axis=-1)/alpha


def sqrt(x):
    return keras.backend.sqrt(x)


def log(x):
    return keras.backend.log(x)


def l2_normalize(x):
    return keras.backend.l2_normalize(x)



input_Tensor=Input(shape=(32,32,3))
model_api = VGG19(input_tensor=input_Tensor, weights='imagenet', include_top=False)

for layers in model_api.layers:
  layers.Traniable=True

feature_vector = model_api.get_layer('block4_conv4').output
#model_api.summary()
mean_vector = keras.layers.Lambda(mean_2)(feature_vector)
reshape_mean_vector= keras.layers.Reshape(target_shape=(4,4,1))(mean_vector)
f1 = keras.layers.Multiply()([reshape_mean_vector,feature_vector])
f2 = keras.layers.Flatten()(f1)
f3 = keras.layers.Dense(256,activation='relu')(f2)
f4 = keras.layers.Dense(128,activation='relu')(f3)
f5 = keras.layers.Dense(10,activation='softmax')(f4)

model = Model(input_Tensor,f5)

from keras.datasets import cifar10
from keras.datasets import cifar100
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import ResNet50
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def DataSet(data):
    if data == 'cifar-10':
        '''
        https://keras.io/api/layers/preprocessing_layers/image_preprocessing/
        '''
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True,
                                     featurewise_center=True,
                                     samplewise_center=True,
                                     featurewise_std_normalization=True,
                                     samplewise_std_normalization=True,
                                     rescale=2,
                                     rotation_range=120)
        
        datagen.fit(x_train)
        datagen.fit(x_test)
        return (x_train,y_train),(x_test,y_test)
    
    elif data == 'cifar-100':
        (x_train,y_train),(x_test,y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True,
                                     featurewise_center=True,
                                     samplewise_center=True,
                                     featurewise_std_normalization=True,
                                     samplewise_std_normalization=True,
                                     rescale=2,
                                     rotation_range=120)
        datagen.fit(x_train)
        datagen.fit(x_test)
        return (x_train,y_train),(x_test,y_test) 
    
    else:
        
        return None
     
def Encoder(model='resnet50'):
    
    if model == 'vgg16':
        input_Tensor = Input(shape=(32,32,3))
        # norm     = Normalization(name='normalization_layer')(input_Tensor)
        ran_ro   = RandomRotation(0.02,seed=0,name='random_rotation')(input_Tensor)
        # ran_hei  =RandomHeight(0.2,seed=0,name='random_height')(ran_ro)
        # rand_wid = RandomWidth(0.2,seed=0,name='random_width')(ran_hei)
        ran_tran = RandomTranslation(height_factor=0.25,width_factor=0.25,seed=0,name='random_translation')(ran_ro)
        rand_flp= RandomFlip('horizontal',name='random_flip',seed=0)(ran_tran)
        ran_con = RandomContrast(factor=0.72,seed=0,name='Random_contrast')(rand_flp)
        model_vgg16 = VGG16(input_tensor=input_Tensor, weights='imagenet', include_top=False)
        inter_layer ='block4_conv3'
        feature_vector = model_vgg16.get_layer(inter_layer).output
        model = Model(input_Tensor,  model_vgg19.output)
        return model
    
    
    elif model == 'vgg19':
        input_Tensor = Input(shape=(32,32,3))
        # norm     = Normalization(name='normalization_layer')(input_Tensor)
        ran_ro   = RandomRotation(0.02,seed=0,name='random_rotation')(input_Tensor)
        # ran_hei  =RandomHeight(0.2,seed=0,name='random_height')(ran_ro)
        # rand_wid = RandomWidth(0.2,seed=0,name='random_width')(ran_hei)
        ran_tran = RandomTranslation(height_factor=0.25,width_factor=0.25,seed=0,name='random_translation')(ran_ro)
        rand_flp= RandomFlip('horizontal',name='random_flip',seed=0)(ran_tran)
        ran_con = RandomContrast(factor=0.72,seed=0,name='Random_contrast')(rand_flp)
        model_vgg19 = VGG19(input_tensor=input_Tensor, weights='imagenet', include_top=False)
        inter_layer ='block4_conv4'
        feature_vector = model_vgg19.get_layer(inter_layer).output
        model = Model(input_Tensor, model_vgg19.output)
        return model
    
    
    elif model == 'resnet50':
        input_Tensor = Input(shape=(32,32,3))
        # # norm     = Normalization(name='normalization_layer')(input_Tensor)
        # ran_ro   = RandomRotation(0.02,seed=0,name='random_rotation')(input_Tensor)
        # # ran_hei  =RandomHeight(0.2,seed=0,name='random_height')(ran_ro)
        # # rand_wid = RandomWidth(0.2,seed=0,name='random_width')(ran_hei)
        # ran_tran = RandomTranslation(height_factor=0.25,width_factor=0.25,seed=0,name='random_translation')(ran_ro)
        # rand_flp= RandomFlip('horizontal',name='random_flip',seed=0)(ran_tran)
        # ran_con = RandomContrast(factor=0.72,seed=0,name='Random_contrast')(rand_flp)
        model_resnet50 = ResNet50(input_tensor= input_Tensor, weights = 'imagenet' , include_top=False)
        # inter_layer='conv3_block4_out'
        # feature_vector = model_resnet50.get_layer(inter_layer).output
        model = Model(input_Tensor, model_resnet50.output)
        return model
    
def build_neural_architecture(attention=True,fv=Encoder(),feature_norm=False,sqrt_norm=True,log_norm=False,l2_norm=True,classes=10):
    
    In = fv
    if attention:
        for layer in In.layers:
            layer.trainable =False
        feature_vector = In.output
        #print(In.summary())
        mean_vector = keras.layers.Lambda(mean_2,name='mean_vector')(feature_vector)
        reshape_mean_vector= keras.layers.Reshape(target_shape=(1,1,1),name='tensor_mean_vector')(mean_vector)
        
        if feature_norm:
            conv = Conv2D(512, kernel_size=(1,1),activation='relu',name='conv_a')(feature_vector)
            conv = Conv2D(512, kernel_size=(1,1),activation='relu',name='conv_b')(conv)
            feature_vector = BatchNormalization(name='batch_norm')(conv)  
            
        
        f1 = keras.layers.Multiply(name='hardamard_prod')([reshape_mean_vector,feature_vector])
        d0 = Dropout(0.4)(f1)
        flat = keras.layers.Flatten(name='flat')(d0)
        
        if sqrt_norm:
            flat = Lambda(sqrt,name='sqrt')(flat)
            
        
        if log_norm:
            flat = Lambda(log,name='log')(flat)
            
            
        if l2_norm:
            flat = Lambda(l2_normalize,name='l2_norm')(flat)
    else:
        for layer in In.layers:
            layer.trainable = True
        feature_vector = In.output
        flat = Flatten()(feature_vector)
    
    f3 = keras.layers.Dense(256,activation='relu',name='dense256')(flat)
    d0 = keras.layers.Dropout(0.5)(f3)
    # f4 = keras.layers.Dense(128,activation='relu', name= 'dense128')(f3)
    # d0 = keras.layers.Dropout(0.5)(f4)
    final = keras.layers.Dense(classes,activation='softmax')(d0)
    model = Model(In.input,final)
    
    return model
    
    




def training(model,x_train,y_train,x_test,y_test):
    model = model
    model.compile(optimizer=Adam(learning_rate=0.0001,amsgrad=True,name='AMSGrad'),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy(name='Acc-T1'),
                           SparseTopKCategoricalAccuracy(k=5,name='Acc-T5')])
    
    batch = 64
    epochs = [130,130,220]
    for i in epochs:
        model.fit(x_train,
                  y_train,
                  epochs=i,
                  batch_size=batch,
                  validation_data=(x_test,
                                   y_test))
        batch+=32
        print('#########--------------Iteration--------------------##########')
    
    


(x_train,y_train),(x_test,y_test) = DataSet(data='cifar-100')
    
nn= build_neural_architecture(fv=Encoder(),attention=True,classes=100)

training(nn,x_train,y_train,x_test,y_test)
    
    
    