from enum import Enum 
from pydantic import BaseModel
from typing import Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class ZMQServices:
    transcription_port:int=11400
   