import asyncio
import aiohttp
import time
from datetime import datetime
from typing import List, Dict
import statistics
import json

# ==================== Configuration ====================
class LoadTestConfig:
    API_URL = "http://localhost:8000/chat"
    TOTAL_REQUESTS = 100
    TIME_WINDOW_SECONDS = 60
    CONCURRENT_USERS = 20  # Number of simulated users
    
# ==================== Test Data ====================
TEST_MESSAGES = [
    "Hello, how are you?",
    "What's the weather like?",
    "Tell me a joke",
    "What can you help me with?",
    "Explain quantum physics",
    "What's your name?",
    "How does AI work?",
    "Tell me about Python programming",
    "What's the meaning of life?",
    "Can you help me with coding?",
    "What time is it?",
    "Tell me something interesting",
    "How do I learn machine learning?",
    "What are your capabilities?",
    "Goodbye!",
]

# ==================== Test Results ====================
class TestResults:
    def __init__(self):
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        
    def add_success(self, response_time: float):
        self.successful_requests += 1
        self.response_times.append(response_time)
    
    def add_failure(self, error: str):
        self.failed_requests += 1
        self.errors.append(error)
    
    def print_summary(self):
        total_requests = self.successful_requests + self.failed_requests
        duration = self.end_time - self.start_time
        
        print("\n" + "="*60)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {self.successful_requests} ({self.successful_requests/total_requests*100:.2f}%)")
        print(f"Failed: {self.failed_requests} ({self.failed_requests/total_requests*100:.2f}%)")
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Requests/Second: {total_requests/duration:.2f}")
        
        if self.response_times:
            print(f"\nResponse Time Statistics:")
            print(f"  Min: {min(self.response_times)*1000:.2f} ms")
            print(f"  Max: {max(self.response_times)*1000:.2f} ms")
            print(f"  Mean: {statistics.mean(self.response_times)*1000:.2f} ms")
            print(f"  Median: {statistics.median(self.response_times)*1000:.2f} ms")
            if len(self.response_times) > 1:
                print(f"  Std Dev: {statistics.stdev(self.response_times)*1000:.2f} ms")
            
            # Percentiles
            sorted_times = sorted(self.response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            print(f"  P95: {sorted_times[p95_idx]*1000:.2f} ms")
            print(f"  P99: {sorted_times[p99_idx]*1000:.2f} ms")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)} total):")
            error_counts = {}
            for error in self.errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
        
        print("="*60 + "\n")

# ==================== Load Testing Functions ====================
async def send_chat_request(
    session: aiohttp.ClientSession,
    user_id: str,
    message: str,
    results: TestResults
) -> Dict:
    """Send a single chat request and record results"""
    start_time = time.time()
    
    try:
        payload = {
            "user_id": user_id,
            "user_input_text": message
        }
        
        async with session.post(
            LoadTestConfig.API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            response_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                results.add_success(response_time)
                return {
                    "success": True,
                    "response_time": response_time,
                    "data": data
                }
            else:
                error_text = await response.text()
                error_msg = f"HTTP {response.status}: {error_text[:100]}"
                results.add_failure(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except asyncio.TimeoutError:
        results.add_failure("Timeout")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        results.add_failure(f"Exception: {type(e).__name__}")
        return {"success": False, "error": str(e)}

async def simulate_user(
    user_id: str,
    num_requests: int,
    results: TestResults,
    delay_between_requests: float = 0
):
    """Simulate a single user making multiple requests"""
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            message = TEST_MESSAGES[i % len(TEST_MESSAGES)]
            
            result = await send_chat_request(session, user_id, message, results)
            
            if result["success"]:
                print(f"✓ User {user_id} - Request {i+1}/{num_requests} - {result['response_time']*1000:.0f}ms")
            else:
                print(f"✗ User {user_id} - Request {i+1}/{num_requests} - Error: {result['error']}")
            
            if delay_between_requests > 0 and i < num_requests - 1:
                await asyncio.sleep(delay_between_requests)

async def run_load_test_concurrent():
    """Run load test with concurrent users"""
    print(f"Starting Load Test: {LoadTestConfig.TOTAL_REQUESTS} requests in {LoadTestConfig.TIME_WINDOW_SECONDS} seconds")
    print(f"Concurrent Users: {LoadTestConfig.CONCURRENT_USERS}")
    print(f"Target API: {LoadTestConfig.API_URL}\n")
    
    results = TestResults()
    results.start_time = time.time()
    
    # Calculate requests per user
    requests_per_user = LoadTestConfig.TOTAL_REQUESTS // LoadTestConfig.CONCURRENT_USERS
    remaining_requests = LoadTestConfig.TOTAL_REQUESTS % LoadTestConfig.CONCURRENT_USERS
    
    # Calculate delay to spread requests over time window
    delay_per_request = LoadTestConfig.TIME_WINDOW_SECONDS / requests_per_user if requests_per_user > 1 else 0
    
    # Create tasks for all users
    tasks = []
    for i in range(LoadTestConfig.CONCURRENT_USERS):
        user_id = f"load_test_user_{i+1}"
        num_reqs = requests_per_user + (1 if i < remaining_requests else 0)
        task = simulate_user(user_id, num_reqs, results, delay_per_request)
        tasks.append(task)
    
    # Run all user simulations concurrently
    await asyncio.gather(*tasks)
    
    results.end_time = time.time()
    results.print_summary()
    
    return results

async def run_load_test_burst():
    """Run load test with all requests sent as fast as possible (burst mode)"""
    print(f"Starting BURST Load Test: {LoadTestConfig.TOTAL_REQUESTS} requests")
    print(f"Concurrent Users: {LoadTestConfig.CONCURRENT_USERS}")
    print(f"Target API: {LoadTestConfig.API_URL}\n")
    
    results = TestResults()
    results.start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(LoadTestConfig.TOTAL_REQUESTS):
            user_id = f"load_test_user_{i % LoadTestConfig.CONCURRENT_USERS + 1}"
            message = TEST_MESSAGES[i % len(TEST_MESSAGES)]
            task = send_chat_request(session, user_id, message, results)
            tasks.append(task)
        
        # Send all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        # Print results
        for i, response in enumerate(responses):
            if response["success"]:
                print(f"✓ Request {i+1} - {response['response_time']*1000:.0f}ms")
            else:
                print(f"✗ Request {i+1} - Error: {response['error']}")
    
    results.end_time = time.time()
    results.print_summary()
    
    return results

async def run_health_check():
    """Check if API is healthy before running tests"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8000/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ API is healthy: {data}\n")
                    return True
                else:
                    print(f"✗ API health check failed: HTTP {response.status}\n")
                    return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}\n")
        print("Make sure the API is running on http://localhost:8000\n")
        return False

# ==================== Main Menu ====================
async def main():
    print("\n" + "="*60)
    print("AI CHATBOT API - LOAD TESTING TOOL")
    print("="*60 + "\n")
    
    # Health check
    if not await run_health_check():
        return
    
    print("Select test mode:")
    print("1. Distributed Load Test (100 requests over 60 seconds)")
    print("2. Burst Test (100 requests as fast as possible)")
    print("3. Custom Test")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        await run_load_test_concurrent()
    elif choice == "2":
        await run_load_test_burst()
    elif choice == "3":
        try:
            LoadTestConfig.TOTAL_REQUESTS = int(input("Total requests: "))
            LoadTestConfig.TIME_WINDOW_SECONDS = int(input("Time window (seconds): "))
            LoadTestConfig.CONCURRENT_USERS = int(input("Concurrent users: "))
            await run_load_test_concurrent()
        except ValueError:
            print("Invalid input. Using default values.")
            await run_load_test_concurrent()
    elif choice == "4":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    asyncio.run(main())