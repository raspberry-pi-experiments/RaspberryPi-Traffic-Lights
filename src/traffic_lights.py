
from argparse import ArgumentParser, ArgumentTypeError
import logging
from input_feeder import InputFeeder
from objects_detection import ObjectsDetection 

class TrafficLights:

    def __init__(self):
        
        self.execute = True

    def parse(self):
        parser = ArgumentParser()
        parser.add_argument('-i', '--input', required=True, type=str,
                            help='select input path to the image or video '
                            'file or specify camera pipeline')
        parser.add_argument(
			'-d', '--debug', type=str, nargs='?', const='DEBUG', default='INFO',
			help="enable debug level (DEBUG by default): NOTSET, DEBUG, INFO, "
			"WARNING, ERROR, CRITICAL")

        return parser

    def run(self, args):
        inputFeeder = InputFeeder(args.input)
        i = 0
        objectsDetection = ObjectsDetection()
        frame = None
        while self.execute:
            try:
                frame = next(inputFeeder.next_batch())
            except StopIteration:
                logging.error('Failed to obtain input stream.')
                break
            if frame is None:
                break
            objectsDetection.inputs(frame)
            objectsDetection.wait()
            outputs = objectsDetection.outputs()
            print(frame)
            i = i + 1
           
        print(i)
        inputFeeder.close()
        

if __name__ == '__main__':
    main = TrafficLights()
    args = main.parse().parse_args()

    logging.basicConfig(
		format="%(asctime)s %(levelname)s: %(message)s",
		level=getattr(logging, args.debug.upper()))

    
    main.run(args)
