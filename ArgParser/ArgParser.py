import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-c", "--classes", nargs="*", help="Classes on which we want to perform detection. If -1, detect all classes.")
parser.add_argument("--conf", type=float,help="Confidence level above which objects will be detected")

# Parse arguments
def parse():
    args = parser.parse_args()
    classes = args.classes
    confidence = args.conf

    intClasses = list()

    if classes is not None:
        for c in classes:
            intClasses.append(int(c))

    d = dict()
    d["classes"] = intClasses
    d["conf"] = confidence
    return d

