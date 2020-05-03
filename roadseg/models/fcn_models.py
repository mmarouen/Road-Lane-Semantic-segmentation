from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Activation, Permute, Reshape, Conv2DTranspose
from keras.models import Model
import cv2
import numpy as np

class FcnModel():
    def __init__(self, n_class, input_height, input_width, model_version=32):
        self.input_height = input_height
        self.input_width = input_width
        self.n_class = n_class
        self.model_version = model_version
        self.weights_path = "weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        self.model = self.build_model()

    def build_model(self):

        img_input = Input(shape=(self.input_height,self.input_width,3))
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_initializer='he_normal')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_initializer='he_normal')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',kernel_initializer='he_normal')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_initializer='he_normal')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',kernel_initializer='he_normal')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',kernel_initializer='he_normal')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',kernel_initializer='he_normal')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',kernel_initializer='he_normal')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',kernel_initializer='he_normal')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',kernel_initializer='he_normal')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',kernel_initializer='he_normal')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',kernel_initializer='he_normal')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',kernel_initializer='he_normal')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        vgg  = Model(img_input , x)
        vgg.load_weights(self.weights_path)

        x = (Conv2D(4096, (7 , 7 ) , activation='relu' , padding='same',kernel_initializer='he_normal',name="conv6"))(x)
        x = Dropout(0.5)(x)
        x = (Conv2D(4096, (1 , 1 ) , activation='relu' , padding='same',kernel_initializer='he_normal',name="conv7"))(x)
        x = Dropout(0.5)(x)

        x = (Conv2D(self.n_class, ( 1 , 1 ),padding='same', kernel_initializer='he_normal', name="scorer1"))(x)
        x = Conv2DTranspose(self.n_class, kernel_size=(64,64), padding='same', strides=(32,32), name="Upsample32")(x)

        o_shape = Model(img_input, x).output_shape
        output_Height = o_shape[1]
        output_Width = o_shape[2]

        x = (Reshape((-1  , output_Height*output_Width)))(x)
        x = (Permute((2, 1)))(x)
        x = (Activation('softmax'))(x)

        model = Model( img_input , x)
        model.outputWidth = output_Width
        model.outputHeight = output_Height
        for l in vgg.layers:
            if "input" not in l.name:
                w=l.get_weights()
                model.get_layer(l.name).set_weights(w)
                model.get_layer(l.name).trainable=False

        return  model

    def predict(self, raw_image, cols=[(0,0,0),(0,255,0)]):
        im_bgd=cv2.resize(raw_image,(self.input_width,self.input_height))
        X = im_bgd.astype(np.float32)
        X = X/255.0
        pr = self.model.predict(np.array([X]))
        pr = pr.reshape((self.input_height,self.input_width, self.n_class)).argmax( axis=2 )
        im_pred = np.zeros((self.input_height,self.input_width, 3))
        for c in range(self.n_class):
            im_pred[:,:,0] += ((pr[:,:] == c )*(cols[c][0])).astype('uint8')
            im_pred[:,:,1] += ((pr[:,:] == c )*(cols[c][1])).astype('uint8')
            im_pred[:,:,2] += ((pr[:,:] == c )*(cols[c][2])).astype('uint8')
        im_pred = cv2.resize(im_pred,(self.input_height,self.input_width))
        im_pred=np.asarray(im_pred,np.uint8)
        return im_bgd,im_pred


