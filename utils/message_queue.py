"""
Message queue integration with RabbitMQ, circuit breaker pattern, and service discovery.
"""

import pika
import time
import json
import threading
import logging
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ServiceInfo:
    """Service information for discovery."""
    name: str
    host: str
    port: int
    status: str
    last_heartbeat: float
    metadata: Dict[str, Any]

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        
    def record_failure(self):
        """Record a failure and update circuit breaker state."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            
    def record_success(self):
        """Record a success and reset circuit breaker."""
        self.failures = 0
        self.state = "CLOSED"
        
    def can_execute(self) -> bool:
        """Check if the operation can be executed."""
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = "HALF-OPEN"
                return True
            return False
            
        return True  # HALF-OPEN state

class MessageQueue:
    """RabbitMQ integration with circuit breaker pattern."""
    
    def __init__(self, host: str = "localhost", port: int = 5672):
        self.host = host
        self.port = port
        self.connection = None
        self.channel = None
        self.circuit_breaker = CircuitBreaker()
        self.service_registry: Dict[str, ServiceInfo] = {}
        self.heartbeat_thread = None
        self.running = True
        
    def connect(self):
        """Establish connection to RabbitMQ with circuit breaker."""
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open")
            
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host, port=self.port)
            )
            self.channel = self.connection.channel()
            self.circuit_breaker.record_success()
            
            # Declare exchange for service discovery
            self.channel.exchange_declare(
                exchange="service_discovery",
                exchange_type="fanout"
            )
            
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._send_heartbeat)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise e
            
    def publish(self, exchange: str, routing_key: str, message: Dict[str, Any]):
        """Publish a message to the queue."""
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open")
            
        try:
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message)
            )
            self.circuit_breaker.record_success()
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise e
            
    def consume(self, queue: str, callback: Callable[[Dict[str, Any]], None]):
        """Consume messages from a queue."""
        def on_message(ch, method, properties, body):
            try:
                message = json.loads(body)
                callback(message)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=on_message
        )
        
    def register_service(self, service_info: ServiceInfo):
        """Register a service in the service registry."""
        self.service_registry[service_info.name] = service_info
        self.publish(
            exchange="service_discovery",
            routing_key="",
            message={
                "type": "SERVICE_REGISTER",
                "service": service_info.__dict__
            }
        )
        
    def discover_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a service by name."""
        return self.service_registry.get(service_name)
        
    def _send_heartbeat(self):
        """Send periodic heartbeat messages."""
        while self.running:
            try:
                for service in self.service_registry.values():
                    service.last_heartbeat = time.time()
                    self.publish(
                        exchange="service_discovery",
                        routing_key="",
                        message={
                            "type": "HEARTBEAT",
                            "service_name": service.name,
                            "timestamp": time.time()
                        }
                    )
                time.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logging.error(f"Error sending heartbeat: {e}")
                
    def close(self):
        """Close the connection and cleanup."""
        self.running = False
        if self.connection and self.connection.is_open:
            self.connection.close()

class ServiceDiscovery(ABC):
    """Abstract base class for service discovery."""
    
    @abstractmethod
    def register_service(self, service_info: ServiceInfo):
        """Register a service."""
        pass
        
    @abstractmethod
    def discover_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a service by name."""
        pass
        
    @abstractmethod
    def update_service_status(self, service_name: str, status: str):
        """Update service status."""
        pass

class RabbitMQServiceDiscovery(ServiceDiscovery):
    """Service discovery implementation using RabbitMQ."""
    
    def __init__(self, message_queue: MessageQueue):
        self.message_queue = message_queue
        
    def register_service(self, service_info: ServiceInfo):
        """Register a service using RabbitMQ."""
        self.message_queue.register_service(service_info)
        
    def discover_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a service using RabbitMQ."""
        return self.message_queue.discover_service(service_name)
        
    def update_service_status(self, service_name: str, status: str):
        """Update service status using RabbitMQ."""
        service = self.message_queue.service_registry.get(service_name)
        if service:
            service.status = status
            self.message_queue.publish(
                exchange="service_discovery",
                routing_key="",
                message={
                    "type": "SERVICE_UPDATE",
                    "service_name": service_name,
                    "status": status
                }
            ) 