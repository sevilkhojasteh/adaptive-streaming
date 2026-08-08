import numpy as np

class ABRSimulator:
    def __init__(self, bitrates, segment_duration=4):
        """
        bitrates: list of available bitrates in kbps
        segment_duration: seconds per segment
        """
        self.bitrates = sorted(bitrates)
        self.segment_duration = segment_duration
        
        self.buffer = 0.0
        self.time = 0.0
        self.rebuffer_time = 0.0
        self.last_bitrate = None
        self.switches = 0
        
        self.history = []

    def rate_based_decision(self, estimated_bandwidth):
        """
        Choose highest bitrate below estimated bandwidth
        """
        candidates = [b for b in self.bitrates if b <= estimated_bandwidth]
        if not candidates:
            return self.bitrates[0]
        return max(candidates)

    def simulate(self, network_trace):
        """
        network_trace: list of bandwidth values (kbps)
        """
        for i, bandwidth in enumerate(network_trace):
            
            # Estimate bandwidth (simple: use current value)
            chosen_bitrate = self.rate_based_decision(bandwidth)
            
            # Segment size in kilobits
            segment_size = chosen_bitrate * self.segment_duration
            
            # Download time (seconds)
            download_time = segment_size / bandwidth
            
            # Playback during download
            if self.buffer > download_time:
                self.buffer -= download_time
            else:
                self.rebuffer_time += (download_time - self.buffer)
                self.buffer = 0
            
            # Add segment to buffer
            self.buffer += self.segment_duration
            
            # Track switching
            if self.last_bitrate is not None and chosen_bitrate != self.last_bitrate:
                self.switches += 1
            
            self.last_bitrate = chosen_bitrate
            
            self.history.append({
                "segment": i,
                "bandwidth": bandwidth,
                "bitrate": chosen_bitrate,
                "buffer": self.buffer
            })
        
        return self.history
