class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.
        [86400, 604800, 2592000]

    From JJ's KTM repository: https://github.com/jilljenn/ktm.
    """
    def __init__(self, window_lengths=None):
        #self.now = None
        self.queue = []
        if window_lengths is not None:
            self.window_lengths = [w for w in window_lengths if w !=float("inf")]
        else:
            self.window_lengths = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
        self.cursors = [0] * len(self.window_lengths)

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)
        return [len(self.queue)] + [len(self.queue) - cursor
                                    for cursor in self.cursors]

    def push(self, time):
        self.queue.append(time)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1
