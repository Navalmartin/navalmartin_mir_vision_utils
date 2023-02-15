
class InvalidPILImageMode(Exception):
    def __init__(self, pil_mode: str):
        self.message = f"The given PIL mode {pil_mode} is invalid"

    def __str__(self):
        return self.message
