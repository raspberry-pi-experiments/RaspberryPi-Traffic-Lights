"""
--------------------------------------------------------------------------------

MIT License

Copyright (c) 2021 Marcin Sielski <marcin.sielski@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------------------------------
"""

# %%
"""
Import all the required modules
"""

from model import Model
import cv2
import numpy as np
import logging
# %%
"""
Define ObjectsDetection class
"""

class ObjectsDetection(Model):

    """
    Class for the facial landmarks detection model
    https://docs.openvinotoolkit.org/2020.1/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
    """

    __MODEL_NAME__ = 'yolo-v2-tiny-ava-0001/FP16/yolo-v2-tiny-ava-0001'


    def __init__(
        self, model_name=__MODEL_NAME__, device='MYRIAD', precision='FP16'):

        """
        Loads selected version of the facial landmarks detection model to the
        selected device

        Note:
            If `device` is set to ``MYRIAD`` or ``GPU`` then `precision` is 
            automatically set to ``FP16`` 

        Args:
            model_name (:obj:`str`, optional): name of the model to load
            device (:obj:`str`, optional): device to infer on
            precision (:obj:`str`, optional): precision of the model to load
        """

        if precision is not 'FP32':
            model_name.replace('FP32', precision)
        if device.startswith('MYRIAD') or device.startswith('MULTI:MYRIAD') or \
            device.startswith('HETERO:MYRIAD') or device.startswith('GPU') or \
                device.startswith('MULTI:GPU') or device.startswith('HETERO:GPU'):
            model_name = 'yolo-v2-tiny-ava-0001/FP16/yolo-v2-tiny-ava-0001'
        super().__init__(self.__MODEL_PATH__ + model_name, device)


    def preprocess_inputs(self, inputs):

        """
        Resizes input image to the size of [3x48x48] and appends it to the list

        Args:
            inputs (ndarray): input image with the human face
        
        Returns:
            list: list of resized images
        """

        self.input = inputs
        input = cv2.resize(self.input, (self._input_shape[3], \
            self._input_shape[2]), interpolation = cv2.INTER_AREA)
        result = []
        result.append(np.moveaxis(input, -1, 0))
        return result


    def preprocess_outputs(self, outputs):

        """
        Extracts images of facial landmarks related to left eye and right eye
        and appends it to the list

        Args:
            outputs (list): list of facial landmarks

        Returns:
            list: list of images with left and right eye
        """

        output = outputs[0].ravel()
        logging.info('Facial landmarks: '+ str(output))
        result = []
        #self.debug = [(int(output[0]*self.input.shape[1]), \
        #    int(output[1]*self.input.shape[0])), \
        #        (int(output[2]*self.input.shape[1]), \
        #            int(output[3]*self.input.shape[0]))]
        #image = self.input[self.debug[0][1]-30:self.debug[0][1]+30,\
        #        self.debug[0][0]-30:self.debug[0][0]+30]
        #result.append(image)
        #image = self.input[self.debug[1][1]-30:self.debug[1][1]+30,\
        #        self.debug[1][0]-30:self.debug[1][0]+30]
        #result.append(image)
        return result
