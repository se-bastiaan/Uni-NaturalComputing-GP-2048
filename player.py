from constants import Move


class Player(object):
    def __init__(self):
        pass
    
    def play(self, game):
        raise NotImplementedError('Implement this in a subclass of Player!')


class DumbPlayer(object):
    did_left = False

    def play(self, game):
        if self.did_left:
            self.did_left = False
            return Move.Down
        else:
            self.did_left = True
            return Move.Left


class GPPlayer(object):
    def __init__(self, individual_fn):
        self.individual_fn = individual_fn

    def play(self, game):
        game = [cell for row in game for cell in row]
        args = {"ARG" + str(i): game[i] for i in range(16)}
        return self.individual_fn(**args)
