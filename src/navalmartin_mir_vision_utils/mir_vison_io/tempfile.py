import uuid
import os


class TempFile:
    def __init__(self, base_path: str = "./", ext: str = ".jpg"):
        self.path = "{base_path}/{rand}{ext}".format(base_path=base_path,
                                                     rand=str(uuid.uuid4()),
                                                     ext=ext)

    def cleanup(self):
        os.remove(self.path)
