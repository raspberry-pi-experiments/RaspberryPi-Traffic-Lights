"""
--------------------------------------------------------------------------------

MIT License

Copyright (c) 2020 Marcin Sielski

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
import cv2

# %%
"""
Define InputFeeder class
"""

class InputFeeder:

    """
    Input feeder class is responsible of delivery of the input images
    """

    def __init__(self, input):

        """
        Initializes images capture from image or video file or from the camera

        Args:
            input (str): path to the image or video file or camera pipeline
        """

        self.cap=cv2.VideoCapture(input)



    def next_batch(self):

        """
        Returns the next image from either a video file or camera.
        """

        while self.cap.isOpened():
            flag, frame = self.cap.read()
            yield frame


    def close(self):

        """
        Closes the VideoCapture
        """

        self.cap.release()
