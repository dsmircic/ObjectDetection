import mimetypes
mimetypes.init()

link = ["https://", "http://"]


def getMediaType(filename: str):
    mimestart = mimetypes.guess_type(filename)[0]

    if mimestart != None:
        mimestart = mimestart.split("/")[0]

        if mimestart == "video":
            return "video"

        elif mimestart == "image":
            return "image"

    else:
        if filename.startswith(link[0]) or filename.startswith(link[1]):
            return "link"

    return "none"
