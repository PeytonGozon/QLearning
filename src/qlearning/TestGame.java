package qlearning;

import game.Game;
import qlearning.AI.QLearningAI;
import qlearning.AI.Utils;
import util.AI;
import util.Context;
import util.GameLoader;
import util.Trial;
import util.model.Model;
import utils.RandomAI;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TestGame {

    final Game ticTacToe = GameLoader.loadGameFromFile(new File("resources/games/tictactoe.lud"));

    // TODO: Ensure to rewrite the QLearningAI without using the tuples. A 3D int[] should suffice
    // TODO: Can also remove the need of passing around the set of moves. All that is needed is the
    // TODO: size of the number of moves.

    // TODO: fix the AI for Amazons. Currently throwing some random Null Pointer Exception
    // TODO: Seems tough to debug..?

    public TestGame() {
        int totalGames = 0, randomAIWins = 0, qLearningAIWins = 0, draws=0;
        final Trial trial = new Trial(ticTacToe);
        final Context context = new Context(ticTacToe, trial);

        final List<AI> ais = new ArrayList<>();
//        QLearningAI qLearningAI = new QLearningAI(0.01, 0.80, 0.10);
        QLearningAI qLearningAI = new QLearningAI(0.01, 0.80, 0.10, "Q-Amazons-0-0-1.bin");
        ais.add(null);
        ais.add(qLearningAI);
        ais.add(new RandomAI());

        final int NUM_TRIALS = 1_000;

        for (int i = 0; i < NUM_TRIALS; ++i)
        {
            ticTacToe.start(context);
            if ((i+1) % 100_000 == 0)
                System.out.println("Game #" + (i+1));

            // Initialize the AIs
            for(int p = 1; p <= ticTacToe.players().count(); ++p)
                ais.get(p).initAI(ticTacToe, p);

            final Model model = context.model();

            while(!trial.over()) {
                model.startNewStep(context, ais, 1.0);

                // Perform an update for the Q-learning AI.
                // In the case of this loop, we will always assign a reward of 0, and handle the winning vs
                // losing below.
            }

//            for(int p = 1; p <= ticTacToe.players().count(); ++p) {
//                AI ai = ais.get(context.state().playerToAgent(p));
//                if (ai == qLearningAI) {
//                    qLearningAI.updateQ(ticTacToe, context, 0);
//                }
//            }

            final double[] ranking = trial.ranking();

            for (int p = 1; p <= ticTacToe.players().count(); ++p) {
//                System.out.println("Agent " + context.state().playerToAgent(p) + " achieved rank: " + ranking[p]);
                AI ai = ais.get(context.state().playerToAgent(p));
                if (ai == qLearningAI) {
                    // We give a reward of 0 for the AI if it draws or for an in-game move.
                    // We give a reward of +1 for the AI should it win.
                    // We give a reward of -1 should the AI lose.
                    int reward = 0;

                    if (ranking[p] == 1) {
                        reward = 1;
                        qLearningAIWins++;
                    } else if (ranking[p] == 2) {
                        reward = -1;
                        randomAIWins++;
                    } else {
                        draws++;
                    }

                    qLearningAI.updateQBackwards(context.game(), context, reward);
                }
            }

            totalGames++;
        }

        System.out.println("Summary: ");
        System.out.println("Q-Learning AI Wins: " + qLearningAIWins + " out of " + totalGames + ". " + 100.0 * qLearningAIWins / totalGames + "% win rate");
        System.out.println("Random Wins: " + randomAIWins + " out of " + totalGames + ". " + 100.0 * randomAIWins / totalGames + "% win rate");
        System.out.println("Draws: " + draws + " out of " + totalGames + ". " + 100.0 * draws / totalGames + "% draw rate");

        System.out.println();
        String AIName = "Q-Amazons-0-0-1.bin";
        System.out.println("Saving AI to: " + AIName);

        try {
            Utils.saveAI(AIName, qLearningAI.getQ());
        } catch (IOException e) {
            System.out.println("Error: Cannot load AI named: " + AIName);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new TestGame();
    }

}
