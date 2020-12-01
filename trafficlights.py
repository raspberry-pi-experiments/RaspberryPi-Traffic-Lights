
from argparse import ArgumentParser, ArgumentTypeError

class TrafficLights:

    def __init__(self):
        pass

    def parse(self):
        parser = ArgumentParser()
        parser.add_argument('-i', '--input', required=True, type=str,
                            help='select input path to the image or video '
                            'file or specify camera pipeline')
        return parser

    def run(self, args):
        pass

if __name__ == '__main__':
    main = TrafficLights()
    args = main.parse().parse_args()
    main.run(args)
