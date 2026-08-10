class QoEModel:
    def __init__(self, alpha=4.0, beta=1.0):
        """
        alpha: rebuffer penalty weight
        beta: switching penalty weight
        """
        self.alpha = alpha
        self.beta = beta

    def compute(self, history, rebuffer_time, switches):
        total_quality = sum([h["bitrate"] for h in history])
        
        qoe = (
            total_quality
            - self.alpha * rebuffer_time * 1000  # convert to bitrate scale
            - self.beta * switches * 1000
        )
        
        return qoe