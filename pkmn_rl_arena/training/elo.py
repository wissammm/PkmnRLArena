import threading

class ELOSystem:
    """ELO rating system for tracking agent performance"""
    def __init__(self, initial_rating=100.0, k_factor=32.0, conf=None):
        self.k_factor = k_factor
        self.ratings = {
            'agent': initial_rating,
            'random': initial_rating,
            'baseline': initial_rating
        }
        self.rating_history = {
            'agent': [initial_rating],
            'random': [initial_rating],
            'baseline': [initial_rating]
        }
        self.match_count = 0
        self.lock = threading.Lock()
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, player_a, player_b, score_a):
        """
        Update ELO ratings after a match
        score_a: 1.0 for A win, 0.5 for draw, 0.0 for A loss
        """
        with self.lock:
            rating_a = self.ratings[player_a]
            rating_b = self.ratings[player_b]
            
            expected_a = self.expected_score(rating_a, rating_b)
            expected_b = 1 - expected_a
            
            new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
            new_rating_b = rating_b + self.k_factor * ((1 - score_a) - expected_b)
            
            self.ratings[player_a] = new_rating_a
            self.ratings[player_b] = new_rating_b
            
            self.rating_history[player_a].append(new_rating_a)
            self.rating_history[player_b].append(new_rating_b)
            
            self.match_count += 1

    def get_rating(self, player):
        with self.lock:
            return self.ratings.get(player, self.conf.initial_elo)
    
    def get_all_ratings(self):
        with self.lock:
            return self.ratings.copy()
    
    def get_rating_history(self):
        with self.lock:
            return {k: list(v) for k, v in self.rating_history.items()}