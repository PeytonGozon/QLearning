package qlearning.learning;

import com.google.common.collect.BiMap;
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
import java.util.List;

public class EpisodicGamePlayer {

    // The underlying game object
    private final Game game;
    final String gameLocation;

    // The name of the Q-Learning AI agent.
    private final String AIName;

    // The number of players in the game.
    private final int numPlayers;

    // Variables for tracking basic statistics, primarily for printing purposes.
    private int numTotalGames = 0, numAI1Wins = 0, numAI2Wins = 0, numDraws = 0;

    /**
     * Constructor and loads a game based off a string.
     * @param gameLocation The location of the game for Ludii to load. Must end in ".lud".
     * @param AIName The name of the underlying AI.
     */
    public EpisodicGamePlayer(final String gameLocation, final String AIName) {
        this.gameLocation = gameLocation;
        // Note: Hex is loaded differently. This will load Hex with a 3x3 grid, but most sizes up to 19x19 are supported.
        if (gameLocation.equals("Hex.lud")) {
            final List<String> options = Arrays.asList("Board Size/3x3");
            this.game = GameLoader.loadGameFromName("Hex.lud", options);
        } else {
            this.game = GameLoader.loadGameFromFile(new File(gameLocation));
        }
        this.AIName = AIName;
        this.numPlayers = game.players().count();
    }

    /**
     * Trains a QLearningAI vs a RandomAI.
     * @param numEpisodes How many episodes of the game to play in the training session.
     * @param switchSidesEachEpisode Whether to switch the AIs's order of their turns.
     * @param alpha the learning rate for the QLearningAI.
     * @param gamma the future reward discount rate for the QLearningAI.
     * @param epsilon the probability of taking a random action for the QLearningAI.
     * @return the number of wins vs the random AI for this training session.
     */
    public double[] performTrainingVSRandomAI(final int numEpisodes, final boolean switchSidesEachEpisode,
                                          final double alpha, final double gamma, final double epsilon,
                                           final boolean outputTrainingData, final int numTimesReport) {
        // Reset the variables for stat tracking.
        numTotalGames = 0;
        numAI1Wins = 0;
        numAI2Wins = 0;
        numDraws = 0;

        // For recording the win percentage of the AI vs the random AI.
        double[] winPercentage = new double[numTimesReport+1];
        int reportIndex = 0;

        // Load the AIs
        final ArrayList<AI> ais = loadAIs("QLearningAI", "Random", alpha, gamma, epsilon);

        // Set up the game
        final Trial trial = new Trial(game);
        final Context context = new Context(game, trial);

        // Determine the maximum width when printing the episodes for aesthetic purposes.
        final int maxWidth = Utils.widthOfNumber(numEpisodes);

        // Get a reference to the Q-learning AI.
        QLearningAI qAI = null;

        for(AI ai : ais)
            if (ai instanceof QLearningAI)
                qAI = (QLearningAI) ai;

        // Perform the training.
        for(int episode = 0; episode < numEpisodes; episode++) {

            // CODE FOR DYNAMICALLY DECREASING EPS
            int l = (int)(numEpisodes * 0.75);
            if (episode <= l) {
                double b = 0;
                double a = 0.5;
                double ratio = episode / (double) l;
                double eps = a * (Math.cos(0.5 * ratio * Math.PI)) + b;
                qAI.setEpsilon(eps);
            } else {
                qAI.setEpsilon(0);
            }
            // END CODE FOR DYNAMICALLY DECREASING EPS

            // Perform one episode of training and determine the rankings.
            final double[] ranking = performOneEpisode(ais, game, trial, context);

            // Reward the Q-learning AI based upon its action.
            rewardAIs(context, ais, ranking);

            // Perform switching the AI if enabled
            if (switchSidesEachEpisode) {
                AI temp = ais.get(1);
                ais.set(1, ais.get(2));
                ais.set(2, temp);
            }

            numTotalGames++;

            // Handle tracking the number of wins.
            if (((numEpisodes > numTimesReport) && (episode % (numEpisodes / numTimesReport)) == 0 && (episode != 0)) || (episode == 500)) {
                System.out.println(String.format("Training Episode #%" + (maxWidth+1) + "d  vs Random AI", episode));
                System.out.println("\tAI1: " + numAI1Wins + "/" + numTotalGames + " = " + 100.0*numAI1Wins/numTotalGames+"%.");
                System.out.println("\tAI2: " + numAI2Wins + "/" + numTotalGames + " = " + 100.0*numAI2Wins/numTotalGames+"%.");
                System.out.println("\tDraws: " + numDraws + "/" + numTotalGames + " = " + 100.0*numDraws/numTotalGames+"%.");
                winPercentage[reportIndex] = (double) numAI1Wins / numTotalGames;
                reportIndex++;
            }
        }

        System.out.println();
        System.out.println("Summary");
        System.out.println("AI1: " + numAI1Wins + "/" + numTotalGames + " = " + 100.0*numAI1Wins/numTotalGames+"%.");
        System.out.println("AI2: " + numAI2Wins + "/" + numTotalGames + " = " + 100.0*numAI2Wins/numTotalGames+"%.");
        System.out.println("Draws: " + numDraws + "/" + numTotalGames + " = " + 100.0*numDraws/numTotalGames+"%.");
        System.out.println();

        if (reportIndex < winPercentage.length)
            winPercentage[reportIndex] = (double) numAI1Wins / numTotalGames;

        // Try to save the Q-Learning AI.
        try {
            Utils.saveAI(AIName + ".bin", qAI.getQ());
        } catch (Exception e) {
            System.err.println("Error: Cannot save the AI.");
        }

        return winPercentage;
    }

    /**
     * Lets the agents play one episode of the game.
     * @param ais an ArrayList of the AIs to play the game.
     * @param local_game A copy of the game.
     * @param trial A copy of the trial.
     * @param context A copy of the context.
     * @return the ranking of the AIs.
     */
    private double[] performOneEpisode(final ArrayList<AI> ais, final Game local_game, final Trial trial, final Context context) {
        // Start the game
        local_game.start(context);

        // Initialize the AIs
        for(int p = 1; p <= this.numPlayers; p++)
            ais.get(p).initAI(local_game, p);

        final Model model = context.model();

        // Perform an episode.
        while (!trial.over())
            model.startNewStep(context, ais, 1.0);

        // Return the rankings of the AIs.
        return trial.ranking();
    }

    /**
     * Rewards the AIs with +1 should they win and -1 should they lose. 0 for draws.
     * @param context The current context.
     * @param ais An arraylist of AIs who are playing the game.
     * @param ranking an array containing the ranking of the AIs from a particular episode.
     */
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

    /**
     * Creates an array list of AIs based upon user-specified values.
     * @param AI1 "QLearningAI" for a Q-learning AI with parameters alpha, gamma, epsilon (that learns). Otherwise, a Random AI.
     * @param AI2 "QLearningAI" for a Q-learning AI with parameters alpha, gamma, epsilon (that learns). Otherwise, a Random AI.
     * @param alpha the Q Learning AI's learning rate
     * @param gamma The Q Learning AI's future reward discount rate
     * @param epsilon The Q Learning AI's epsilon greedy policy parameter.
     * @return an array list of AIs. Note: initAI has not been called yet.
     */
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
