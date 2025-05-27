"""
Async utilities for ELVIS Trading Bot
Provides async processing capabilities for improved performance
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Callable, Optional, TypeVar, Coroutine
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncTaskManager:
    """Manage async tasks with rate limiting and error handling"""
    
    def __init__(self, max_concurrent: int = 10, rate_limit: Optional[int] = None):
        """
        Initialize async task manager
        
        Args:
            max_concurrent: Maximum concurrent tasks
            rate_limit: Maximum requests per second (None for no limit)
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = AsyncRateLimiter(rate_limit) if rate_limit else None
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def execute_tasks(self, tasks: List[Coroutine[Any, Any, T]]) -> List[T]:
        """
        Execute multiple async tasks concurrently
        
        Args:
            tasks: List of coroutines to execute
            
        Returns:
            List of results in the same order as tasks
        """
        async def execute_with_semaphore(task):
            async with self.semaphore:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                try:
                    return await task
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    return None
        
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=False
        )
        
        return results
    
    async def fetch_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch URL asynchronously
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            Response data
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        async with self.semaphore:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            
            try:
                async with self._session.get(url, **kwargs) as response:
                    return {
                        'status': response.status,
                        'data': await response.json(),
                        'headers': dict(response.headers)
                    }
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return {'error': str(e)}
    
    async def batch_fetch(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch multiple URLs concurrently
        
        Args:
            urls: List of URLs to fetch
            **kwargs: Additional arguments for aiohttp requests
            
        Returns:
            List of response data
        """
        tasks = [self.fetch_url(url, **kwargs) for url in urls]
        return await self.execute_tasks(tasks)


class AsyncRateLimiter:
    """Rate limiter for async operations"""
    
    def __init__(self, rate: int):
        """
        Initialize rate limiter
        
        Args:
            rate: Maximum operations per second
        """
        self.rate = rate
        self.period = 1.0 / rate
        self.last_call = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to maintain rate limit"""
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call
            
            if time_since_last < self.period:
                sleep_time = self.period - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_call = time.time()


class AsyncBatchProcessor:
    """Process items in batches asynchronously"""
    
    def __init__(self, batch_size: int = 100, process_func: Optional[Callable] = None):
        """
        Initialize batch processor
        
        Args:
            batch_size: Size of each batch
            process_func: Function to process each batch
        """
        self.batch_size = batch_size
        self.process_func = process_func
        self.queue = asyncio.Queue()
        self.results = []
        self._running = False
    
    async def add_item(self, item: Any):
        """Add item to processing queue"""
        await self.queue.put(item)
    
    async def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch of items"""
        if self.process_func:
            if asyncio.iscoroutinefunction(self.process_func):
                return await self.process_func(batch)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.process_func, batch)
        return batch
    
    async def start(self):
        """Start processing batches"""
        self._running = True
        
        while self._running:
            batch = []
            
            # Collect items for batch
            try:
                # Wait for at least one item
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                batch.append(item)
                
                # Try to fill the batch
                while len(batch) < self.batch_size and not self.queue.empty():
                    try:
                        item = self.queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break
                
                # Process batch if we have items
                if batch:
                    results = await self.process_batch(batch)
                    self.results.extend(results)
                    
            except asyncio.TimeoutError:
                # No items in queue, continue waiting
                continue
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    async def stop(self):
        """Stop processing"""
        self._running = False
        
        # Process remaining items
        if not self.queue.empty():
            batch = []
            while not self.queue.empty():
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            
            if batch:
                results = await self.process_batch(batch)
                self.results.extend(results)


def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for async function retry logic
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


class AsyncCache:
    """Simple async cache with TTL"""
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize async cache
        
        Args:
            default_ttl: Default time to live in seconds
        """
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() < entry['expires']:
                    return entry['value']
                else:
                    del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        expires = time.time() + ttl
        
        async with self.lock:
            self.cache[key] = {
                'value': value,
                'expires': expires
            }
    
    async def delete(self, key: str):
        """Delete value from cache"""
        async with self.lock:
            self.cache.pop(key, None)
    
    async def clear(self):
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
    
    async def cleanup(self):
        """Remove expired entries"""
        async with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time >= entry['expires']
            ]
            for key in expired_keys:
                del self.cache[key]


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine from sync context
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)
    else:
        # Running loop exists, use thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


async def gather_with_concurrency(n: int, *coros):
    """
    Gather coroutines with limited concurrency
    
    Args:
        n: Maximum concurrent coroutines
        *coros: Coroutines to execute
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*(sem_coro(c) for c in coros))
