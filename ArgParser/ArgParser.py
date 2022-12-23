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



def parse():
    args = parser.parse_args()
    classes = args.classes
    confidence = args.conf
    source = args.source
    dest = args.dest

    intClasses = list()

    if classes is not None:
        for c in classes:
            intClasses.append(int(c))

    d = dict()
    d["classes"] = intClasses
    d["conf"] = confidence
    d["source"] = source
    d["dest"] = dest
    return d
