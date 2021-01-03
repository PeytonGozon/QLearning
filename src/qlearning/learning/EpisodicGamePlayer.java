package qlearning.learning;

import game.Game;
import qlearning.AI.QLearningAI;
import qlearning.util.Utils;
import util.AI;
import util.Context;
import util.GameLoader;
import util.Trial;
import util.model.Model;
import utils.RandomAI;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

public class EpisodicGamePlayer {

    /*
    What do I need?

    - # of training episodes vs random
    - # of training episodes vs self
    - # of times to do self play
    - Do we swap roles each time?

    - Some structure to contain the two AIs
    - Some way to switch their position during the learning process? for self play purposes
    - A way to save the progress of the current AI
     */

    private final Game game;
    private final String AIName;
    private final int numPlayers;
    private int numTotalGames = 0, numAI1Wins = 0, numAI2Wins = 0, numDraws = 0;


    public EpisodicGamePlayer(final String gameLocation, final String AIName) {
        this.game = GameLoader.loadGameFromFile(new File(gameLocation));
        this.AIName = AIName;
        this.numPlayers = game.players().count();
    }

    /**
     * Trains a QLearningAI vs a RandomAI.
     * @param numEpisodes
     * @param switchSidesEachEpisode
     * @param alpha the learning rate for the QLearningAI.
     * @param gamma the future reward discount rate for the QLearningAI.
     * @param epsilon the probability of taking a random action for the QLearningAI.
     */
    public void performTrainingVSRandomAI(final int numEpisodes, final boolean switchSidesEachEpisode,
                                          final double alpha, final double gamma, final double epsilon) {
        // Reset the variables for stat tracking.
        numTotalGames = 0;
        numAI1Wins = 0;
        numAI2Wins = 0;
        numDraws = 0;

        // Load the AIs
        final ArrayList<AI> ais = loadAIs("QLearningAI", "Random", alpha, gamma, epsilon);

        // Set up the game
        final Trial trial = new Trial(game);
        final Context context = new Context(game, trial);

        final int maxWidth = Utils.widthOfNumber(numEpisodes);

        QLearningAI qAI = null;

        for(AI ai : ais)
            if (ai instanceof QLearningAI)
                qAI = (QLearningAI) ai;

        // Perform the training.
        for(int episode = 0; episode < numEpisodes; episode++) {

            // TEST CODE FOR DYNAMICALLY DECREASING EPS
            int l = 100_000;
            if (episode <= l) {
                double b = 0;
                double a = 0.5;
                double ratio = episode / (double) l;
                double eps = a * (Math.cos(0.5 * ratio * Math.PI)) + b;
                qAI.setEpsilon(eps);
            } else if(episode > l) {
                qAI.setEpsilon(0);
            }

            // END TEST CODE FOR DYNAMICALLY DECREASING EPS


            if ((numEpisodes > 10) && (episode % (numEpisodes / 10)) == 0) {
//                System.out.println(String.format("Training Episode #%" + maxWidth + "d  vs Random AI", episode));
//                System.out.println("\tAI1: " + numAI1Wins + "/" + numTotalGames + " = " + 100.0*numAI1Wins/numTotalGames+"%.");
//                System.out.println("\tAI2: " + numAI2Wins + "/" + numTotalGames + " = " + 100.0*numAI2Wins/numTotalGames+"%.");
//                System.out.println("\tDraws: " + numDraws + "/" + numTotalGames + " = " + 100.0*numDraws/numTotalGames+"%.");
            }

            final double[] ranking = performOneEpisode(ais, trial, context);

            rewardAIs(context, ais, ranking);

            // Perform switching the AI if enabled
            if (switchSidesEachEpisode) {
                AI temp = ais.get(1);
                ais.set(1, ais.get(2));
                ais.set(2, temp);
            }

            numTotalGames++;
        }

        System.out.println();
        System.out.println("AI1: " + numAI1Wins + "/" + numTotalGames + " = " + 100.0*numAI1Wins/numTotalGames+"%.");
        System.out.println("AI2: " + numAI2Wins + "/" + numTotalGames + " = " + 100.0*numAI2Wins/numTotalGames+"%.");
        System.out.println("Draws: " + numDraws + "/" + numTotalGames + " = " + 100.0*numDraws/numTotalGames+"%.");
        System.out.println();



        try {
            Utils.saveAI(AIName + ".bin", qAI.getQ());
        } catch (Exception e) {
            System.err.println("Error: Cannot save the AI.");
        }
    }

    private double[] performOneEpisode(final ArrayList<AI> ais, final Trial trial, final Context context) {
        // Start the game
        game.start(context);

        // Initialize the AIs
        for(int p = 1; p <= this.numPlayers; p++)
            ais.get(p).initAI(game, p);

        final Model model = context.model();

        // Perform an episode.
        while (!trial.over())
            model.startNewStep(context, ais, 1.0);

        // Return the rankings of the AIs.
        return trial.ranking();
    }

    private void rewardAIs(final Context context, final ArrayList<AI> ais, final double[] ranking) {
        if(ais.size() != this.numPlayers+1)
            System.err.println("Error: the number of AIs is not equal to the number of players of the game!");

        if(ranking.length != this.numPlayers+1)
            System.err.println("Error: the number of rewards is not equal to the number of players of the game!");

        // update Stat tracking variables.
        int winner = 0;

        // Reward each AI.
        for (int p = 1; p <= this.numPlayers; ++p) {
            AI ai = ais.get(context.state().playerToAgent(p));
            if (ranking[p] == 1.0) winner = p;
            // If the AI is a QLearningAI, then we reward it +1 should it win, -1 should it lose, and 0 for a draw.
            if (ai instanceof QLearningAI) {
                int reward = 0;
                if (ranking[p] == 1) {
                    reward = 1;
                } else if (ranking[p] == 2) {
                    reward = -1;
                }
                ((QLearningAI) ai).updateQBackwards(reward);
            }
        }

        if (winner == 1)
            numAI1Wins++;
        else if (winner == 2)
            numAI2Wins++;
        else
            numDraws++;
    }


    private ArrayList<AI> loadAIs(final String AI1, final String AI2, final double alpha,
                         final double gamma, final double epsilon) {
        // Indexing begins with 1 in Ludii for AIs, so a null AI is added to the beginning.
        ArrayList<AI> ais = new ArrayList<>();
        ais.add(null);
        ais.add(AI1.equals("QLearningAI") ? new QLearningAI(alpha, gamma, epsilon, true) : new RandomAI());
        ais.add(AI2.equals("QLearningAI") ? new QLearningAI(alpha, gamma, epsilon, true) : new RandomAI());
        return ais;
    }

}
