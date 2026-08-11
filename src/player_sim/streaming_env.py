import numpy as np

class StreamingEnv:
    """
    RL environment for adaptive bitrate streaming.
    State:  normalized [throughput_mbps, buffer_sec/10, last_bitrate_ratio]
    Action: index into self.bitrates
    Reward: Pensieve-style, per-step (small bounded values)
    """
    def __init__(self, trace, bitrates, segment_duration=4, max_buffer=60.0):
        self.trace = trace
        self.bitrates = sorted(bitrates)
        self.segment_duration = segment_duration
        self.max_buffer = max_buffer
        self.max_bitrate = max(self.bitrates)
        self.reset()

    def reset(self):
        self.ptr = 0
        self.buffer = 0.0
        self.rebuffer_time = 0.0        # cumulative, for logging only
        self.last_bitrate = self.bitrates[0]
        self.done = False
        return self._get_state()

    def step(self, action_index):
        if self.done:
            return self._get_state(), 0.0, True

        bandwidth      = self.trace[self.ptr]                       # kbps
        chosen_bitrate = self.bitrates[action_index]                # kbps

        # Download this segment
        segment_size  = chosen_bitrate * self.segment_duration      # kilobits
        download_time = segment_size / max(bandwidth, 1e-6)         # seconds

        # Play from buffer while downloading
        if self.buffer >= download_time:
            self.buffer -= download_time
            step_rebuffer = 0.0
        else:
            step_rebuffer = download_time - self.buffer
            self.buffer   = 0.0

        # Segment finished downloading → add to buffer
        self.buffer = min(self.buffer + self.segment_duration, self.max_buffer)

        # ---------- Pensieve-style reward (PER-STEP, in log units) ----------
        # Use log-quality (your original idea was good, just applied wrong)
        quality       = np.log(chosen_bitrate / self.bitrates[0])   # >= 0
        rebuf_penalty = 4.3 * step_rebuffer                          # ← STEP rebuffer, not total
        smooth_pen    = abs(
            np.log(chosen_bitrate / self.bitrates[0])
            - np.log(self.last_bitrate / self.bitrates[0])
        )
        reward = quality - rebuf_penalty - smooth_pen
        # -------------------------------------------------------------------

        # bookkeeping
        self.rebuffer_time += step_rebuffer
        self.last_bitrate   = chosen_bitrate
        self.ptr           += 1
        if self.ptr >= len(self.trace):
            self.done = True

        return self._get_state(), reward, self.done

    def _get_state(self):
        if self.ptr >= len(self.trace):
            bandwidth = 0.0
        else:
            bandwidth = self.trace[self.ptr]

        # ---------- NORMALIZED state (all roughly O(1)) ----------
        return np.array([
            bandwidth / 1000.0,                        # throughput in Mbps
            self.buffer / 10.0,                        # buffer in ~[0, 6]
            self.last_bitrate / self.max_bitrate,      # bitrate ratio [0, 1]
        ], dtype=np.float32)