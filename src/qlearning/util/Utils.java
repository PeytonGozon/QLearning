package qlearning.util;

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
     * Requires 1-9 players.
     * @param context the current episode's context.
     * @return a (unique) identifier for the board.
     */
    public static int boardToHashcode(final Context context) {
        final Board currentBoard = context.board();
        final Trial currentTrial = context.trial();
        final int numSites = currentBoard.numSites();
        // double the length of the number of legal sites, in an attempt to make a unique hashcode.
        Integer[] initialBoard = new Integer[numSites];
        Arrays.fill(initialBoard, 0);

        Iterator<Move> iterator = currentTrial.reverseMoveIterator();

        while (iterator.hasNext()) {
            Move m = iterator.next();
            int where = m.to();
            int playerID = m.what();
            // If where is -1, then the move is a pass, and may be ignored.
            if (where != -1)
                initialBoard[where] = playerID;
        }

//        return (new String(initialBoard)).hashCode();
        return Arrays.deepHashCode(initialBoard);
    }

    public static void saveAI(final String fileName, final ConcurrentHashMap<Integer, double[]> Q)
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

    public static ConcurrentHashMap<Integer, double[]> loadAI(final String fileName) {
        ConcurrentHashMap<Integer, double[]> Q = null;

        final String pathName = "resources/AIs/" + fileName;

        try {
            FileInputStream fis = new FileInputStream(pathName);
            ObjectInputStream in = new ObjectInputStream(fis);
            Q = (ConcurrentHashMap<Integer, double[]>)in.readObject();
        } catch (IOException | ClassNotFoundException ex) {
            System.err.println("Error: Could not load Q from " + pathName + ". Aborting.");
            ex.printStackTrace();
        }

        return Q;
    }

    public static int widthOfNumber(final int num) {
        int width = 0;
        int temp = num;
        while(temp / 10 > 0) {
            width++;
            temp /=10;
        }
        return width;
    }

}
