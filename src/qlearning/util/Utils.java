package qlearning.util;

import com.google.common.collect.BiMap;
import game.equipment.container.board.Board;
import util.Context;
import util.Move;
import util.Trial;

import java.io.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

public class Utils {

    /**
     * Converts a board state into a unique hashcode. Requires 1-9 players.
     * @param context the current episode's context.
     * @return a (unique) identifier for the board.
     */
    public static int boardToHashcode(final Context context) {
        final Board currentBoard = context.board();
        final Trial currentTrial = context.trial();
        final int numSites = currentBoard.numSites();
        // double the length of the number of legal sites to ensure that Arrays.deepHashCode results in a
        // unique hashcode for a reasonable board size.
        Integer[] initialBoard = new Integer[2*numSites];
        Arrays.fill(initialBoard, 0);

        // Encode the board into the first half.
        Iterator<Move> iterator = currentTrial.reverseMoveIterator();
        while (iterator.hasNext()) {
            Move m = iterator.next();
            int where = m.to();
            int playerID = m.what();
            // If where is -1, then the move is a pass, and may be ignored.
            if (where != -1)
                initialBoard[where] = playerID;
        }

        // Fill the second half of the array with less trivial numbers to better ensure the uniqueness
        // of the hashcode.
        for(int i = 0; i < numSites; i++)
            initialBoard[i + numSites] = 331319 * initialBoard[i];

        // Utilize the deepHashCode to determine a unique hash code for an array of integers to be independent of
        // their location in memory.
        return Arrays.deepHashCode(initialBoard);
    }

    /**
     * Given the underlying Q factors of a Q-learning AI, save it uniquely to a file.
     * @param fileName The name and extension of the AI, which will be stored in "/resources/AI/". .
     * @param Q The Bidirectional Q map for a given Q-Learning AI.
     * @throws IOException if unable to save the Q-factors to the given file.
     */
    public static void saveAI(final String fileName, final BiMap<Integer, double[]> Q)
        throws IOException
    {
        final String pathName = "resources/AIs/" + fileName;
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(pathName))) {
            out.writeObject(Q);
        } catch(IOException e) {
            System.err.println("Error: Could not write Q to " + pathName + ". Aborting.");
            e.printStackTrace();
        }
    }

    /**
     * Loads the Q-factors of a Q-learning AI from a given file.
     * @param fileName The name and extension of the AI loaded from "/resources/AI/".
     * @return The stored BiDirectional map.
     */
    public static BiMap<Integer, double[]> loadAI(final String fileName) {
        BiMap<Integer, double[]> Q = null;

        final String pathName = "resources/AIs/" + fileName;

        try {
            FileInputStream fis = new FileInputStream(pathName);
            ObjectInputStream in = new ObjectInputStream(fis);
            Q = (BiMap<Integer, double[]>)in.readObject();
        } catch (IOException | ClassNotFoundException ex) {
            System.err.println("Error: Could not load Q from " + pathName + ". Aborting.");
            ex.printStackTrace();
        }

        return Q;
    }

    /**
     * A helper function for determining how many digits long a number is, which is used for printing.
     */
    public static int widthOfNumber(final int num) {
        int width = 0;
        int temp = num;
        while (temp / 10 > 0) {
            width++;
            temp /= 10;
        }
        return width;
    }

}
