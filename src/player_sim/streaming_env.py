import numpy as np

class StreamingEnv:
    def __init__(self, trace, bitrates, segment_duration=4):
        self.trace = trace
        self.bitrates = bitrates
        self.segment_duration = segment_duration
        
        self.reset()

    def reset(self):
        self.ptr = 0
        self.buffer = 0
        self.rebuffer_time = 0
        self.last_bitrate = self.bitrates[0]
        self.done = False
        
        return self._get_state()

    def step(self, action_index):
        if self.done:
            return self._get_state(), 0, True

        bandwidth = self.trace[self.ptr]
        chosen_bitrate = self.bitrates[action_index]

        segment_size = chosen_bitrate * self.segment_duration
        download_time = segment_size / bandwidth

        # playback during download
        if self.buffer > download_time:
            self.buffer -= download_time
        else:
            self.rebuffer_time += (download_time - self.buffer)
            self.buffer = 0

        self.buffer += self.segment_duration

        # reward (log quality model)
        reward = (
            np.log(chosen_bitrate)
            - 4.0 * self.rebuffer_time
            - abs(np.log(chosen_bitrate) - np.log(self.last_bitrate))
        )

        self.last_bitrate = chosen_bitrate

        self.ptr += 1
        if self.ptr >= len(self.trace):
            self.done = True

        return self._get_state(), reward, self.done

    def _get_state(self):
        if self.ptr >= len(self.trace):
            bandwidth = 0
        else:
            bandwidth = self.trace[self.ptr]

        return np.array([
            bandwidth,
            self.buffer,
            self.last_bitrate
        ], dtype=np.float32)
