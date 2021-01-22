package qlearning;

import main.FileHandling;
import qlearning.learning.EpisodicGamePlayer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.List;

public class Main {

    public static void main(String[] args) {

        // the Q-learning AI model parameters. If usingDynamicEps is false, then epsilon is just a+b.
        final double alpha = 0.1;  // the learning rate.
        final double gamma = 0.9;  // the discounted future reward ratio.
        final double a = 0.5;      // a+b is the upper bound of the epsilon-greedy policy parameter.
        final double b = 0;        // Lower bound for the epsilon-greedy policy parameter.
        final int m = 45_000;      // The number of episodes during each trial.
        final int l = 30_000;      // Which episode to stop exploring.
        final boolean usingDynamicEps = true;
        // The number of times to perform training and evaluating the model.
        final int NUM_BATCHES = 3;
        // How many times to tick marks to put into the data for the purpose of creating visualizations.
        final int REPORT_EVERY = 3_000;
        // The number of episodes during each trial. This is `m`.
        // Which game to play
        final String gameName = "tictactoe";



        // The name of the AI and corresponding CSV file containing information
        final String AIName = gameName + "-" + m + "-alpha" + alpha + "-gamma" +
                gamma + "-a" + a + "-b" + b + "-l" + l + "-usingDynamicEps " + usingDynamicEps;
        final String csvName = "CSVs/" + AIName + ".csv";

        String gameLocation = "resources/games/" + gameName + ".lud";
        for (String name : FileHandling.listGames()) {
            final String adjustedName = gameName + ".lud";
            if (name.contains(adjustedName)) {
                gameLocation = adjustedName;
                break;
            }
        }

        final Path csvFilePath = Paths.get(csvName);

        // Create the game player object, which handles the facilitation of AIs and playing the game.
        EpisodicGamePlayer gamePlayer = new EpisodicGamePlayer(gameLocation, AIName);


        // Create the CSV file that stores how well the model has performed and append headers
        try {
            StringBuilder headers = new StringBuilder();
            for(int i = 1; i <= (m / REPORT_EVERY); i++)
                headers.append(i * REPORT_EVERY).append(",");

            List<String> contents = Collections.singletonList(headers.toString());
            Files.write(csvFilePath, contents, StandardCharsets.UTF_8, StandardOpenOption.CREATE);
        } catch (Exception ex) {
            System.err.println("Error: could not create file " + csvName + ".");
            ex.printStackTrace();
        }

        // A string builder to store the results of the model
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < NUM_BATCHES; i++) {
            // Create a model that plays for NUM_EPISODES, and has alpha = 0.1, gamma = 0.9, and epsilon_0 = 0.50.
            double[] winPercentage = gamePlayer.performTrainingVSRandomAI(m, l,
                    alpha, gamma, a+b, a, b, REPORT_EVERY, false, usingDynamicEps);

            for (double v : winPercentage) {
                builder.append(v).append(",");
            }
            builder.append("\n");
        }

        try {
            List<String> contents = Collections.singletonList(builder.toString());
            Files.write(csvFilePath, contents, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        } catch (Exception ex) {
            System.err.println("Error: could not open file " + csvName + " and write.");
            ex.printStackTrace();
        }
    }
}
