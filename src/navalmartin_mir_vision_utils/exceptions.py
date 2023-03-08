
class InvalidPILImageMode(Exception):
    def __init__(self, pil_mode: str):
        self.message = f"The given PIL mode {pil_mode} is invalid"

    def __str__(self):
        return self.message


class InvalidConfiguration(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self) -> str:
        return self.message
