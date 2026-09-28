import psutil


def monitor_worker(shared_cpu, shared_ram, stop_event, interval=0.5):
    """
    Background process to monitor system resources without blocking
    the main application loop.

    Uses stop_event.wait(timeout) instead of time.sleep so the process
    wakes immediately when stop_event is set rather than waiting up to
    interval seconds after shutdown is signalled.
    """
    print("[Monitor] Starting system monitoring...")
    while not stop_event.wait(timeout=interval):
        shared_cpu.value = psutil.cpu_percent(interval=None)
        shared_ram.value = psutil.virtual_memory().percent
    print("[Monitor] Stopping system monitoring.")


def get_system_usage(interval=0.2):
    return psutil.cpu_percent(interval=interval), psutil.virtual_memory().percent
