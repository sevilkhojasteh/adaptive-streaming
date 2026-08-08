import numpy as np

class QoEModel:
    def __init__(self, alpha=4.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def compute(self, history, rebuffer_time):
        bitrates = [h["bitrate"] for h in history]

        quality = sum(np.log(bitrates))

        smoothness = sum(
            abs(np.log(bitrates[i]) - np.log(bitrates[i-1]))
            for i in range(1, len(bitrates))
        )

        qoe = quality - self.alpha * rebuffer_time - self.beta * smoothness
        return qoe