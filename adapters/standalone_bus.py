"""
Standalone data bus for non-ROS2 operation.
Simulates ROS-like pub/sub with queues.
"""

from typing import Dict, Callable, Any, Optional
from queue import Queue
from threading import Thread, Lock

from ..core.logger import setup_logger

logger = setup_logger(__name__)


class StandaloneBus:
    """
    Simple pub/sub bus for standalone operation.
    """
    
    def __init__(self, queue_maxsize: int = 10):
        """
        Initialize standalone bus.
        
        Args:
            queue_maxsize: Maximum queue size for each topic
        """
        self.queue_maxsize = queue_maxsize
        self.topics: Dict[str, Queue] = {}
        self.subscribers: Dict[str, list] = {}
        self.lock = Lock()
    
    def publish(self, topic: str, message: Any):
        """
        Publish message to topic.
        
        Args:
            topic: Topic name
            message: Message data
        """
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = Queue(maxsize=self.queue_maxsize)
            
            # Add to queue
            try:
                if self.topics[topic].full():
                    self.topics[topic].get_nowait()  # Remove oldest
                self.topics[topic].put_nowait(message)
            except Exception as e:
                logger.warning(f"Error publishing to {topic}: {e}")
            
            # Call subscribers
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback for {topic}: {e}")
    
    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe to topic.
        
        Args:
            topic: Topic name
            callback: Function to call with message
        """
        with self.lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
            logger.debug(f"Subscribed to {topic}")
    
    def get_latest(self, topic: str) -> Optional[Any]:
        """
        Get latest message from topic without blocking.
        
        Args:
            topic: Topic name
            
        Returns:
            Latest message or None if queue empty
        """
        with self.lock:
            if topic not in self.topics:
                return None
            
            queue = self.topics[topic]
            latest = None
            
            # Drain queue to get latest
            while not queue.empty():
                try:
                    latest = queue.get_nowait()
                except:
                    break
            
            return latest
