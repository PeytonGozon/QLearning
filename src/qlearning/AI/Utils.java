package qlearning.AI;

import util.Context;
import util.Move;
import util.Trial;

import java.io.*;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

public class Utils {

    public static int boardToHashcode(final Context context) {
        final Trial currentTrial = context.trial();
        Iterator<Move> iterator = currentTrial.reverseMoveIterator();
        int hashcode = 0;
        while (iterator.hasNext()) {
            Move m = iterator.next();
            hashcode += m.what() * Math.pow(10, m.to());
        }

        return hashcode;
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

}
