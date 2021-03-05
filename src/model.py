'''
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
'''

# %%
'''
Import all the required modules
'''
from openvino.inference_engine import IENetwork, IECore
import logging
# %%
'''
Define Model base class
'''

class Model:

    '''
    Model class is base class for all the models. It defines common methods to
    use model
    '''

    __MODEL_PATH__ = '../model/intel/'

    def __init__(self, model_name, device='CPU'):

        '''
        Loads selected model to the selected device

        Args:
            model_name (str): name of the model to load
            device (:obj:`str`, optional): device to infer on
        '''

        self._model_weights = model_name+'.bin'
        self._model_structure = model_name+'.xml'
        self._device = device
        
        core = IECore()
        try:
            self._model=core.read_network(self._model_structure, self._model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. "\
                "Have you enterred the correct model path?")

        self._input_names = []
        for input_name in iter(self._model.inputs):
            self._input_names.append(input_name)
        self._input_shape = self._model.inputs[self._input_names[0]].shape
        self._output_names = []
        for output_name in iter(self._model.outputs):
            self._output_names.append(output_name)
        #try:
        #    supported_layers = core.query_network(network=self._model, \
        #        device_name=device)
        #    unsupported_layers = \
        #        [l for l in self._model.layers.keys() if l not in #supported_layers]
        #    if len(unsupported_layers) != 0:
        #        logging.warning("Unsupported layers found: {}".format#(unsupported_layers))
        #except RuntimeError:
        #    logging.error('Failed to query device: ' + self._device)
        #    exit(1)
        try:
            self._network = core.load_network(network=self._model, \
                device_name=self._device, num_requests=1)
        except RuntimeError:
            logging.error('Failed to load model on: ' + self._device)
            exit(1)


    def preprocess_inputs(self, inputs):

        '''
        Abstract method used to preprocess inputs

        Args:
            inputs: inputs to preprocess
        '''

        raise NotImplementedError


    def preprocess_outputs(self, outputs):

        '''
        Abstract method to preprocess outputs

        Args:
            outputs: outputs to preprocess
        '''

        raise NotImplementedError


    def inputs(self, inputs):

        '''
        Starts asynchronous inference on the inputs

        Args:
            inputs (list): list of input data to infer on
        '''

        data = self.preprocess_inputs(inputs)
        inputs = {}
        for input_name, input in zip(self._input_names, data):
            inputs[input_name] = input
        self._network.start_async(request_id=0, inputs=inputs)


    def wait(self):

        '''
        Waits for asynchronous inference result

        Returns:
            int: inference result
        '''

        return self._network.requests[0].wait(-1)


    def outputs(self):

        '''
        Returns the results of the inference

        Returns:
            list: list of inference results
        '''

        outputs = []
        for output in self._output_names:
            outputs.append(self._network.requests[0].outputs[output])
        return self.preprocess_outputs(outputs)
