import mimetypes
mimetypes.init()

link = ["https://", "http://"]


def get_media_type(filename: str):
    """
    Determines which type of media is the file specified with "filename."

    Parameters
    ----------
    filename:
        name of the file for which we want to determine the media type.
    """

    if filename == "0":
        return "camera"

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
