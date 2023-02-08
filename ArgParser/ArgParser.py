import argparse

"""
Parses arguments from the command line.
"""
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-c", "--classes", nargs="*",
                    help="Classes on which we want to perform detection. If -1, detect all classes.")
parser.add_argument("--conf", type=float,
                    help="Confidence level above which objects will be detected")
parser.add_argument("--source", type=str,
                    help="Data on which detection will be made.")
parser.add_argument("--dest", type=str,
                    help="Name of the file where the detection result will be stored.")
parser.add_argument("--speed", type=int, default=5,
                    help="Speed of the detection, 1 is the slowest, 10 is the fastest. Determines how many frames will be skipped before each detection. Default is 5. ")
parser.add_argument("--base", type=int, default=-1,
                    help="Object which has to be detected in order to detect other objects which are on top of it.")
parser.add_argument("--overlap", type=int, default=0.5,
                    help="Area of intersection between the base object and the object on top of it.")


def parse():
    args = parser.parse_args()
    classes = args.classes
    confidence = args.conf
    source = args.source
    dest = args.dest
    speed = args.speed
    base = args.base
    overlap = args.overlap

    intClasses = list()

    if classes is not None:
        for c in classes:
            intClasses.append(int(c))

    d = dict()
    d["classes"] = intClasses
    d["conf"] = confidence
    d["source"] = source
    d["dest"] = dest
    d["speed"] = speed
    d["base"] = base
    d["overlap"] = overlap
    return d
