from abc import ABC, abstractmethod
from typing import Dict, Any
from graph.state import GraphState


class Node(ABC):
    """
    Abstract class for a node in the graph
    """

    @abstractmethod
    def action(state: GraphState) -> Dict[str, Any]:
        pass
