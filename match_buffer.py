

class PredMatchBuffer:
    def __init__(self):
        self.match = []
        self.score = []

    def add(self, match, score):
        self.match.append(match)
        self.score.append(score)