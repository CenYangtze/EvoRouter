import sys
import random
import threading
from typing import Union, Literal, List


class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        self.value = 0.0


class Cost(Singleton):
    def __init__(self):
        self._local = threading.local()

    @property
    def value(self):
        return getattr(self._local, "value", 0.0)

    @value.setter
    def value(self, value):
        self._local.value = float(value)


class PromptTokens(Singleton):
    def __init__(self):
        self._local = threading.local()

    @property
    def value(self):
        return getattr(self._local, "value", 0.0)

    @value.setter
    def value(self, value):
        self._local.value = float(value)


class CompletionTokens(Singleton):
    def __init__(self):
        self._local = threading.local()

    @property
    def value(self):
        return getattr(self._local, "value", 0.0)

    @value.setter
    def value(self, value):
        self._local.value = float(value)


class Time(Singleton):
    def __init__(self):
        self.value = ""


class Mode(Singleton):
    def __init__(self):
        self.value = ""
