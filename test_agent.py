import torch
import numpy as np

import snake_game
from agent import Agent
import cv2


def test_game():
    game = snake_game.SnakeGameAI(food_number=1)
    agent = Agent()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    agent.model.load_state_dict(torch.load("./model/Policy_ConvNet4.pth", map_location=torch.device('cpu')))
    # agent.model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))
    agent.model.eval()
    agent.model.to(device)  # Move our model to the gpu memory

    game_over = False
    while not game_over:
        # get state
        # state = agent.get_state_pixels(game)
        state = agent.get_state(game)

        # get move
        final_move = agent.get_action(state, game)

        # perform move and get new state
        reward, done, score = game.play_step(agent.move_prediction2str(final_move),
                                             visuals=True,
                                             food_number=1)

        # print(reward, game_over, score, final_move, agent.move_prediction2str(final_move))
        cv2.waitKey(5)

        if done:
            # train long memory, plot result
            game.reset()


def test_state():
    agent = Agent()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    agent.model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))
    agent.model.eval()
    agent.model.to(device)  # Move our model to the gpu memory

    state = cv2.imread('test_state.jpg')
    state = np.transpose(state, (2, 0, 1))  # conv

    # get move
    final_move = [0, 0, 0, 0]  # [Right, Up, Left, Down]
    state_tensor = torch.tensor(state, dtype=torch.float).to(device)
    state_tensor = state_tensor.unsqueeze(0)  # conv

    prediction = agent.model(state_tensor)
    move = torch.argmax(prediction).item()
    final_move[move] = 1
    print(prediction.shape)


if __name__ == "__main__":
    test_game()
    # test_state()
