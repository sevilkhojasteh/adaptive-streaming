import numpy as np

class ABRSimulator:
    def __init__(self, bitrates, segment_duration=4, ema_alpha=0.5):
        self.bitrates = sorted(bitrates)
        self.segment_duration = segment_duration
        self.ema_alpha = ema_alpha
        
        self.reset()

    def reset(self):
        self.buffer = 0.0
        self.rebuffer_time = 0.0
        self.switches = 0
        self.last_bitrate = self.bitrates[0]
        self.estimated_bandwidth = None
        self.history = []

    # -------------------------
    # Rate-Based ABR
    # -------------------------
    def rate_based_decision(self):
        candidates = [b for b in self.bitrates if b <= self.estimated_bandwidth]
        if not candidates:
            return self.bitrates[0]
        return max(candidates)

    # -------------------------
    # Buffer-Based ABR
    # -------------------------
    def buffer_based_decision(self):
        if self.buffer < 4:
            return self.bitrates[0]
        elif self.buffer < 8:
            return self.bitrates[1]
        elif self.buffer < 12:
            return self.bitrates[2]
        elif self.buffer < 16:
            return self.bitrates[3]
        else:
            return self.bitrates[-1]

    # -------------------------
    # Simulation
    # -------------------------
    def simulate(self, network_trace, mode="rate"):

        self.reset()

        for i, bandwidth in enumerate(network_trace):

            # Estimate bandwidth using EMA
            if self.estimated_bandwidth is None:
                self.estimated_bandwidth = bandwidth
            else:
                self.estimated_bandwidth = (
                    self.ema_alpha * bandwidth
                    + (1 - self.ema_alpha) * self.estimated_bandwidth
                )

            # Choose bitrate
            if mode == "rate":
                chosen_bitrate = self.rate_based_decision()
            elif mode == "buffer":
                chosen_bitrate = self.buffer_based_decision()
            else:
                raise ValueError("Unknown mode")

            segment_size = chosen_bitrate * self.segment_duration
            download_time = segment_size / bandwidth

            # Playback during download
            if self.buffer > download_time:
                self.buffer -= download_time
            else:
                self.rebuffer_time += (download_time - self.buffer)
                self.buffer = 0

            self.buffer += self.segment_duration

            # Count switches
            if chosen_bitrate != self.last_bitrate:
                self.switches += 1

            self.last_bitrate = chosen_bitrate

            self.history.append({
                "bandwidth": bandwidth,
                "estimated_bandwidth": self.estimated_bandwidth,
                "bitrate": chosen_bitrate,
                "buffer": self.buffer
            })

        return self.history