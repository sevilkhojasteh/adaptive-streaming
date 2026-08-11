import numpy as np

class QoEModel:
    """
    Pensieve-style QoE:
        QoE = Σ q(R_n) − α·rebuffer − β·Σ|q(R_n) − q(R_{n-1})|
    where q(R) can be linear (R/1000) or logarithmic (log(R/R_min)).
    """
    def __init__(self, alpha=4.3, beta=1.0, mode="log"):
        self.alpha = alpha
        self.beta  = beta
        self.mode  = mode

    def _q(self, bitrate, min_bitrate):
        if self.mode == "log":
            return np.log(bitrate / min_bitrate)
        return bitrate / 1000.0     # linear (Mbps)

    def compute(self, history, rebuffer_time, switches=None):
        bitrates = [h["bitrate"] for h in history]
        min_br   = min(bitrates)

        # 1) total quality
        quality_sum = sum(self._q(b, min_br) for b in bitrates)

        # 2) smoothness (recompute from history so old callers still work)
        smooth_sum = sum(
            abs(self._q(bitrates[i], min_br) - self._q(bitrates[i-1], min_br))
            for i in range(1, len(bitrates))
        )

        qoe = quality_sum - self.alpha * rebuffer_time - self.beta * smooth_sum
        return qoe