class GameState:
    def __init__(self, tank1, tank2, bullets):
        self.tank1 = tank1
        self.tank2 = tank2
        self.bullets = bullets

    def done(self):
        for bullet in self.bullets:
            if bullet.x == self.tank1.x and bullet.y == self.tank1.y:
                return True
            if bullet.x == self.tank2.x and bullet.y == self.tank2.y:
                return True
        return False

    def get_legal_actions(self, agent):
        if agent == 1:
            return self.tank1.get_legal_actions()
        elif agent == 2:
            return self.tank2.get_legal_actions()

    def apply_action_1(self, action):
        # if action starts with 'move', move the tank
        if action.startswith('MOVE'):
            self.tank1.move(action)
        # if action starts with 'shoot', shoot a bullet
        elif action.startswith('SHOOT'):
            self.tank2.shoot(action)
        for bullet in self.bullets:
            if bullet.moves > 0:
                bullet.move()
            else:
                bullet.moves += 1
            if bullet.moves >= 10:
                self.bullets.remove(bullet)

    def apply_action_2(self, action):
        if action.startswith('MOVE'):
            self.tank2.move(action)
        elif action.startswith('SHOOT'):
            self.tank2.shoot(action)
        for bullet in self.bullets:
            if bullet.moves > 0:
                bullet.move()
            else:
                bullet.moves += 1
            if bullet.moves >= 10:
                self.bullets.remove(bullet)

    def generate_successor(self, agent, action):
        successor = GameState(self.tank1, self.tank2, self.bullets)
        if agent == 1:
            successor.apply_action_1(action)
        elif agent == 2:
            successor.apply_action_2(action)
        return successor
