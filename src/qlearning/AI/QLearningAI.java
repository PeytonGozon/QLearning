package qlearning.AI;

import game.Game;
import main.collections.FastArrayList;
import qlearning.Tuple;
import util.AI;
import util.Context;
import util.Move;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadLocalRandom;

public class QLearningAI extends AI {

    // Player ID
    protected int player = -1;

    // Learning rate, discount rate, epsilon-greedy policy parameter
    private final double alpha, gamma, epsilon;

    // Whether for the AI to learn or not.
    private final boolean learn;

    // The underlying Q table.
    private static volatile ConcurrentHashMap<Integer, double[]> Q = null;

    // Move History (for this episode)
    // Stores a list of Board Hashes, Move Made.
//    private static volatile ConcurrentLinkedQueue<Tuple<Integer, Integer, FastArrayList<Move>>> moveHistory = null;
    private static volatile LinkedBlockingDeque<Tuple<Integer, Integer, FastArrayList<Move>>> moveHistory = null;

    /**
     * Constructor (for creating a new agent)
     * @param alpha the learning rate of the model
     * @param gamma the future reward discount rate
     * @param epsilon the probability the model takes a random move via an epsilon-greedy policy.
     * @param train whether the model should keep learning from playing.
     */
    public QLearningAI(double alpha, double gamma, double epsilon, boolean train) {
        this.friendlyName = "Q-Learning Agent";
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.learn = train;
    }

    /**
     * Constructor (for creating a new agent that learns by default)
     * @param alpha the learning rate of the model
     * @param gamma the future reward discount rate
     * @param epsilon the probability the model takes a random move via an epsilon-greedy policy.
     */
    public QLearningAI(double alpha, double gamma, double epsilon) {
        this(alpha, gamma, epsilon, true);
    }

    /**
     * Constructor (for loading a model with continual training)
     * @param alpha the learning rate of the model
     * @param gamma the future reward discount rate
     * @param epsilon the probability the model takes a random move via an epsilon-greedy policy.
     * @param modelName the name of the model to load, located in resources/AIs/modelName.
     * @param train whether the model should keep learning from playing.
     */
    public QLearningAI(double alpha, double gamma, double epsilon, String modelName, boolean train) {
        this(alpha, gamma, epsilon, train);

        if (Q == null)
            Q = Utils.loadAI(modelName);
    }

    /**
     * Constructor (for loading a model without future training)
     * @param alpha the learning rate of the model
     * @param gamma the future reward discount rate
     * @param epsilon the probability the model takes a random move via an epsilon-greedy policy.
     * @param modelName the name of the model to load, located in resources/AIs/modelName.
     */
    public QLearningAI(double alpha, double gamma, double epsilon, String modelName) {
        this(alpha, gamma, epsilon, modelName,false);
    }

    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;

        if (Q == null)
            Q = new ConcurrentHashMap<>();
        if(moveHistory == null)
            moveHistory = new LinkedBlockingDeque<>();
    }

    @Override
    public Move selectAction(final Game game,
                             final Context context,
                             final double maxSeconds,
                             final int maxIterations,
                             final int maxDepth) {

        // Deep copy the allowed moves.
        FastArrayList<Move> legalMoves = new FastArrayList<>();
        legalMoves.addAll(game.moves(context).moves());

        // Pre-emptively obtain a legal random move.
        final int numLegalMoves = legalMoves.size();

        final int randomLegalMove = ThreadLocalRandom.current().nextInt(numLegalMoves);

        assert randomLegalMove < legalMoves.size() : "Invalid move size";

        // Determine the board's hash code
        final int boardHashCode = Utils.boardToHashcode(context);

        // The final move we choose to make
        final int moveChoice;

        // Perform an Epsilon-Greedy Policy for choosing a move.
        if (ThreadLocalRandom.current().nextDouble(0, 1) < this.epsilon)
        {
            // We will return the random move.
            moveChoice = randomLegalMove;
        } else {
            // Otherwise, we select the optimal list in memory at this state.
            // Note: If there are no optimal moves from this state yet, then we will
            // choose the random move.

            // Determine the possible Q values from this state.
            final double[] QValues = this.getQValues(legalMoves, boardHashCode);

            // Find the arg max Q value. The maximum value's index is our optimal move choice.
            final int maxQIndex = argmax(QValues);

            moveChoice = QValues[maxQIndex] > 0 ? maxQIndex : randomLegalMove;
        }

        // Finally, return the optimal move, as defined by the policy.
        // Also store the move to the move history.
        Move selectedMove = legalMoves.get(moveChoice);

        // Ensure that the move choice is legal
//        assert moveChoice < legalMoves.size();

        // Put the moves in backwards order to the Queue, so that it is in FILO order.
        moveHistory.addFirst(new Tuple<>(boardHashCode, moveChoice, legalMoves));

        return selectedMove;
    }

    public void updateQBackwards(final Game game, final Context context, double reward) {
        // Implement a backwards Queue that allows for the whole game to be played
        // And then, once the game is over, retroactively go down the queue to apply
        // Q updates in place. This way, a whole game is simulated at once before updates occur.
        // and are retroactively accounted for via the chain nature.

        // So, we assume that the whole game has been played, with a reward of zero assumed for all but
        // the very ending state.
        if (moveHistory == null) {
            System.err.println("Error: moveHistory was not instantiated. Aborting.");
            return;
        } else if (moveHistory.isEmpty()) {
            System.err.println("Error: moveHistory is empty when attempting to update the Q values. Aborting.");
            return;
        }

        Iterator<Tuple<Integer, Integer, FastArrayList<Move>>> queueIterator = moveHistory.iterator();

        if(!queueIterator.hasNext())
        {
            System.err.println("Error: moveHistory does not have any next nodes at the start of processing. Aborting.");
            return;
        }

        //
        // Obtain the first node in the queue (the last state of the game).
        //
        // These will be updated as the queue progresses to store the previous state, and hence are not final.
        //
        // As we are effectively stepping backwards through the queue, we begin with the most recent moves
        // and then update the move before it.
        //
        Tuple<Integer, Integer, FastArrayList<Move>> currentMove = queueIterator.next();
        int currentBoardHashcode = currentMove.getItem1();
        FastArrayList<Move> currentLegalMoves = currentMove.getItem3();

        while (queueIterator.hasNext()) {
            // Decode the current node from the queue.
            final Tuple<Integer, Integer, FastArrayList<Move>> previousMove = queueIterator.next();
            final int previousBoardHashcode = previousMove.getItem1();
            final int previousMoveChoice = previousMove.getItem2();
            final FastArrayList<Move> previousLegalMoves = previousMove.getItem3();

            final double[] currentQValuesArray = getQValues(currentLegalMoves, currentBoardHashcode);

            // Find the optimal Q value of the current step. This will be used to update the Q value
            // of the previous state.
            final double maxCurrentQValue = max(currentQValuesArray);

            //
            // Perform the Q-learning update.
            //
            double[] previousQValuesArray = getQValues(previousLegalMoves, previousBoardHashcode);
            final double initialQValue = previousQValuesArray[previousMoveChoice];
            final double updatedQValue = initialQValue + this.alpha * (reward + this.gamma * maxCurrentQValue - initialQValue);

            // Update the array of previous Q Values, for memory efficiency.
            previousQValuesArray[previousMoveChoice] = updatedQValue;

            // Update Q.
            Q.put(previousBoardHashcode, previousQValuesArray);

            // Set reward to be zero, as only the very final state of the game receives a potentially non-zero reward.
            reward = 0;

            // Update the current parameters.
            currentBoardHashcode = previousBoardHashcode;
            currentLegalMoves = previousLegalMoves;
        }

        // Reset the move history.
        moveHistory.clear();
    }


    public double[] getQValues(final FastArrayList<Move> legalMoves, final int boardHashCode) {
        final double[] qValues;

        if (Q == null) throw new AssertionError("Error: Q must be initialized. ");

        if (!Q.containsKey(boardHashCode)) {
            qValues = new double[legalMoves.size()];
            Q.put(boardHashCode, qValues);
        } else {
            qValues = Q.get(boardHashCode);
        }

        return qValues;
    }

    /**
     * Returns the largest element within the array. If the array is ewmpty, it returns
     * the largest negative value a double may contain.
     * @param array the list of doubles.
     * @return the largest element of the array.
     */
    public double max(double[] array) {
        if (array == null) throw new AssertionError("Error: array must not be null.");

        double maxValue = -(Double.MAX_VALUE - 1);

        for (double currentQValue : array)
            if (maxValue < currentQValue)
                maxValue = currentQValue;
        return maxValue;
    }

    /**
     * Returns the index of the maximum element within the array. If the array is empty, it returns
     * the largest negative value of a double.
     * @param array the list of doubles.
     * @return the index of the largest element.
     */
    public int argmax(double[] array) {

        if (array == null) throw new AssertionError("Error: array must not be null.");

        double maxValue = -(Double.MAX_VALUE - 1);
        int argMax = -1;
        for (int i = 0; i < array.length; i++) {
            if(maxValue < array[i]) {
                argMax = i;
                maxValue = array[i];
            }
        }
        return argMax;
    }

    public ConcurrentHashMap<Integer, double[]> getQ() {
        return Q;
    }

}
