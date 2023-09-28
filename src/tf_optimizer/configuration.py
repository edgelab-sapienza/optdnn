from configparser import ConfigParser
import re


class Configuration:
    def __init__(self) -> None:
        self.config = ConfigParser()
        self.config.read("config.ini")

    @staticmethod
    def convert_to_types(input_string):
        if not isinstance(input_string, str):
            return input_string

        if input_string.lower() == "true":
            return True
        elif input_string.lower() == "false":
            return False

        int_pattern = re.compile(r"^[+-]?\d+$")
        float_pattern = re.compile(r"^[+-]?\d+(\.\d+)?$")

        if int_pattern.match(input_string):
            as_int = int(input_string)
            return as_int
        elif float_pattern.match(input_string):
            as_float = float(input_string)
            return as_float
        return input_string

    def getConfig(self, familiy: str, value: str) -> any:
        v = self.config[familiy][value]
        return Configuration.convert_to_types(v)
