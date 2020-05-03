from keras.models import Model
from keras.layers import Cropping2D

def crop(a,b,input_image):
  
  o_shape1 = Model(input_image, a ).output_shape
  outputHeight1 = o_shape1[1]
  outputWidth1 = o_shape1[2]

  o_shape2 = Model(input_image , b).output_shape
  outputHeight2 = o_shape2[1]
  outputWidth2 = o_shape2[2]
    
  cx = abs( outputWidth1 - outputWidth2 )
  cy = abs( outputHeight2 - outputHeight1 )
  
  if outputWidth1 > outputWidth2:
    a = Cropping2D(cropping=((0,0) , ( 0 , cx )))(a)
  else:
    b = Cropping2D(cropping=((0,0) , ( 0 , cx )))(b)
    
  if outputHeight1 > outputHeight2 :
    a = Cropping2D(cropping=((0,cy) , ( 0 , 0 )))(a)
  else:
    b = Cropping2D(cropping=((0, cy ) , ( 0 , 0)))(b)
  return a , b
