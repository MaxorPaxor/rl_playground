import numpy as np
import cv2
import time

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
FPS = 22


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.frame = np.zeros((self.h, self.w, 3))
        # init game state
        self.reset()

    def reset(self):
        # reset
        self.direction = 'right'
        self.head = [int(self.w / 2), int(self.h / 2)]
        self.snake = [self.head.copy(),
                     [self.head[0] - BLOCK_SIZE, self.head[1]],
                     [self.head[0] - (2 * BLOCK_SIZE), self.head[1]]]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = np.random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = np.random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = [x, y]

        if self.food in self.snake:
            self._place_food()

    def get_action_human(self):
        action = None
        time_start = time.time()
        dt = 0

        while dt < 1 / FPS:
            move = self.direction
            k = cv2.waitKey(1)
            # Right = 83, up = 82, left = 81, down = 84, q = 113
            if k == 81:
                move = 'left'
                other = 'right'
            if k == 82:
                move = 'up'
                other = 'down'
            if k == 83:
                move = 'right'
                other = 'left'
            if k == 84:
                move = 'down'
                other = 'up'
            if k == ord('q'):
                action = 'quit'

            if k != -1 and move != self.direction and other != self.direction:
                # clicked = True
                action = move

            time_end = time.time()
            dt = time_end - time_start

        return action

    def play_step(self, action=None, visuals=False):
        self.frame = np.zeros((self.h, self.w, 3))
        self.frame_iteration += 1

        # 1. collect user input there is no policy action
        if action is None:
            action = self.get_action_human()
            if action == 'quit':
                return 0, True, self.score

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head.copy())

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui
        if visuals:
            # Draw food
            self.frame = cv2.rectangle(self.frame,
                                       (self.food[0], self.food[1]),
                                       (self.food[0] + BLOCK_SIZE, self.food[1] + BLOCK_SIZE),
                                       RED, thickness=-1)
            # Draw snake
            for link in self.snake:
                self.frame = cv2.rectangle(self.frame,
                                           (int(link[0]), int(link[1])),
                                           (int(link[0] + BLOCK_SIZE), int(link[1] + BLOCK_SIZE)),
                                           WHITE, thickness=-1)

            cv2.imshow('Snake', self.frame)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] < 0 or pt[1] > self.h - BLOCK_SIZE:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # [straight, right, left]

        new_dir = None

        if self.direction == 'right':
            if action == 'up':
                new_dir = 'up'
            if action == 'down':
                new_dir = 'down'
        if self.direction == 'up':
            if action == 'left':
                new_dir = 'left'
            if action == 'right':
                new_dir = 'right'
        if self.direction == 'left':
            if action == 'up':
                new_dir = 'up'
            if action == 'down':
                new_dir = 'down'
        if self.direction == 'down':
            if action == 'left':
                new_dir = 'left'
            if action == 'right':
                new_dir = 'right'

        if new_dir != self.direction and new_dir is not None:
            self.direction = new_dir

        # Move the head
        if self.direction == 'right':
            self.head[0] += BLOCK_SIZE
        elif self.direction == 'left':
            self.head[0] -= BLOCK_SIZE
        elif self.direction == 'down':
            self.head[1] += BLOCK_SIZE
        elif self.direction == 'up':
            self.head[1] -= BLOCK_SIZE


if __name__ == "__main__":
    game = SnakeGameAI()
    game_over = False
    while not game_over:
        rew, game_over, score = game.play_step(visuals=True)

    cv2.destroyAllWindows()
    print("Final score: {}".format(score))
