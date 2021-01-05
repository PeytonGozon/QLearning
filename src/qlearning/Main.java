package qlearning;

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

        // The number of times to perform training and evaluating the model.
        final int NUM_BATCHES = 100;
        // How many times to tick marks to put into the data for the purpose of creating visualizations.
        final int NUM_TIMES_REPORT = 10;
        // The number of episodes during each trial.
        final int NUM_EPISODES = 200_000;
        // Which game to play
        final String gameLocation = "resources/games/tictactoe.lud";


        // The name of the AI and corresponding CSV file containing information
        final String AIName = "tictactoe";
        final String csvName = AIName + ".csv";
        final Path csvFilePath = Paths.get(csvName);

        // Create the game player object, which handles the facilitation of AIs and playing the game.
        EpisodicGamePlayer gamePlayer = new EpisodicGamePlayer(gameLocation, AIName);


        // Create the CSV file that stores how well the model has performed and append headers
        try {
            StringBuilder headers = new StringBuilder();
            headers.append("Batch 500,");
            for(int i = 1; i <= NUM_TIMES_REPORT; i++)
                headers.append("Batch ").append(i * (NUM_EPISODES / NUM_TIMES_REPORT)).append(",");

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
            double[] winPercentage = gamePlayer.performTrainingVSRandomAI(NUM_EPISODES, false,
                    0.1, 0.9, 0.50, true, NUM_TIMES_REPORT);

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
