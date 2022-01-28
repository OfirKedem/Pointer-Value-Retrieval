from .vectorized import VectorPVR
from .visual_block_style import BlockStylePVR

tasks_registry = {'VectorPVR': VectorPVR,
                  'BlockStylePVR': BlockStylePVR}
