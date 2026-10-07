import timeit
import time

class Timer:
    def __init__(self):
        self.start_time_total = None
        self.start_time_process = None
        self.acc_time_total = 0.0
        self.acc_time_process = 0.0

    def start(self):
        if self.start_time_total is None or self.start_time_process is None:
            self.start_time_total = timeit.default_timer()
            self.start_time_process = time.process_time()

    def pause(self):
        if self.start_time_total is None or self.start_time_process is None:
            return  # has already been paused or not been started
        self.acc_time_total += timeit.default_timer() - self.start_time_total
        self.acc_time_process += time.process_time() - self.start_time_process
        self.start_time_total = None
        self.start_time_process = None

    def get_result_dict(self):
        return {'total': self.acc_time_total, 'process': self.acc_time_process}
