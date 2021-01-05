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
        final String AIName = "tictactoe";
        final int NUM_TIMES_REPORT = 10;
        final int NUM_EPISODES = 200_000;

        EpisodicGamePlayer gamePlayer = new EpisodicGamePlayer("resources/games/tictactoe.lud",
                AIName);

        final String csvName = AIName + ".csv";
        final Path csvFilePath = Paths.get(csvName);

        // Create the file and append headers
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

        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < 25; i++) {
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
