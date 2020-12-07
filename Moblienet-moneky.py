from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Activation,Flatten
from keras.models import Sequential,Model
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D,Conv2D,ZeroPadding2D

img_rows,img_cols=224,224

MobileNet=MobileNet(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))


##not including the top layers
for layer in MobileNet.layers:
    layer.trainable=False
    
    
for(i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

    
## till now we donot have the head for the model,as the incluse_top=False

def addTopModellayer(bottom_model,num_classes):
    top_model=bottom_model.output
    ## here the top model is added to the bottom model
    top_model=GlobalAveragePooling2D()(top_model)## another way of adding the layers to the model
    top_model=Dense(1024,activation='relu')(top_model)
    top_model=Dense(1024,activation='relu')(top_model)
    top_model=Dense(512,activation='relu')(top_model)
    top_model=Dense(num_classes,activation='softmax')(top_model)
    return top_model

## completed the adding of the layers to the top model and addded to the bottom  model




num_classes=10 ## According to the dataset

FC_Head=addTopModellayer(MobileNet,num_classes)


model= Model(inputs=MobileNet.input,outputs=FC_Head)

print(model.summary())

##Load the dataset


from keras.preprocessing.image import ImageDataGenerator


#data augumentation

train_data_dir = './train'
validation_data_dir = './validation'


## data  generation for the train and valdiadation set. 
train_datagen = ImageDataGenerator(
      rescale=1./255,rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint,EarlyStopping

ModelCheckpoint=ModelCheckpoint("Monkey/data.h5",
                                monitor="val_loss",
                                mode='min',
                                save_best_only=True,
                                verbose=1)



EarlyStopping=EarlyStopping(monitor='val_loss',patience='2',verbose=1,
                            restore_best_weights=True,min_delta=0)

callbacks=[ModelCheckpoint,EarlyStopping]

model.compile(loss="categorical_crossentropy",optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])

nb_train_samples=1097
nb_validation_samples=272



epochs=5
batch_size=16

history=model.fit_generator(train_generator,steps_per_epoch=nb_train_samples,
                            callbacks=callbacks,validation_data=validation_generator,
                            validation_steps=nb_validation_samples)






















