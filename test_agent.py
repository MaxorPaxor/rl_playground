import torch
import numpy as np

import snake_game
from agent import Agent
import cv2


def test_game():
    game = snake_game.SnakeGameAI(food_number=3)
    agent = Agent()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    agent.model.load_state_dict(torch.load("./model/model_conv.pth", map_location=torch.device('cpu')))
    agent.model.eval()
    agent.model.to(device)  # Move our model to the gpu memory

    game_over = False
    while not game_over:
        # get state
        state = agent.get_state_pixels(game)
        # cv2.imwrite('test_state.jpg', state)
        # state = cv2.imread('test_state.jpg')
        state = np.transpose(state, (2, 0, 1))  # conv

        # get move
        final_move = [0, 0, 0, 0]  # [Right, Up, Left, Down]
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        state_tensor = state_tensor.unsqueeze(0)  # conv

        prediction = agent.model(state_tensor)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        # perform move and get new state
        reward, game_over, score = game.play_step(action=agent.move_list2str(final_move),
                                                  visuals=True,
                                                  food_number=3)

        print(reward, game_over, score, prediction.shape, agent.move_list2str(final_move))
        cv2.waitKey(20)


def test_state():
    agent = Agent()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    agent.model.load_state_dict(torch.load("./model/model_conv.pth", map_location=torch.device('cpu')))
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
