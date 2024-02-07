from enum import IntEnum


class ProcessErrorCode(IntEnum):
    InputShapeNotDetectable = 101
    WrongQuantizationType = 102
    ConnectionRefused = 103
    LowAccuracy = 104
