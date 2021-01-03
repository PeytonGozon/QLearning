package qlearning;

import qlearning.learning.EpisodicGamePlayer;
import qlearning.util.Utils;

import java.util.concurrent.ConcurrentHashMap;

public class Main {

    public static void main(String[] args) {
        for(int i = 0; i < 10; i++) {
            EpisodicGamePlayer gamePlayer = new EpisodicGamePlayer("resources/games/tictactoe5.lud",
                    "Q-AI-tictactoe5-0 0 1");


            gamePlayer.performTrainingVSRandomAI(200_000, false, 0.1,
                    0.9, 0.50);
        }
    }
}
