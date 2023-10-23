from enum import IntEnum


class ProcessErrorCode(IntEnum):
    InputShapeNotDetectable = -1
    WrongQuantizationType = -2
    ConnectionRefused = -3
