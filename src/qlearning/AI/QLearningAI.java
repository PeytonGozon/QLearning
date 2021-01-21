package qlearning.AI;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import game.Game;
import main.collections.FastArrayList;
import org.jetbrains.annotations.NotNull;
import qlearning.util.Utils;
import util.AI;
import util.Context;
import util.Move;

import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ThreadLocalRandom;

public class QLearningAI extends AI {

    // Player ID
    protected int player = -1;

    // Learning rate, discount rate, epsilon-greedy policy parameter
    private final double alpha, gamma;
    private double epsilon;

    // Whether for the AI to learn or not.
    private final boolean learn;

    // The underlying Q table.
    private volatile BiMap<Integer, double[]> Q = null;

    // Move History (for this episode)
    // Stores a list of Board Hashes, Move Made.
    // Storage order: board hashcode, the move choice made, and the number of legal moves at the time.
    private volatile ConcurrentLinkedDeque<int[]> moveHistory = null;

    /**
     * A default constructor used when loading this AI From the Ludii platform. This requires a file
     * "resources/AIs/Q-AI-0 0 1.bin" in to exist relative to the jar.
     */
    public QLearningAI() {
        this(0.01, 0.80, 0, "Q-AI-0 0 1.bin", false);
    }

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
     * initAI is a default method in the base.AI class that should be called prior to the AI being trained.
     * @param game the Game object.
     * @param playerID the unique ID assigned to this AI
     */
    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;

        this.Q = HashBiMap.create();
        if(moveHistory == null)
            moveHistory = new ConcurrentLinkedDeque<>();
    }

    @Override
    public Move selectAction(final Game game,
                             final Context context,
                             final double maxSeconds,
                             final int maxIterations,
                             final int maxDepth) {

        // Obtain a set of legal moves
        final @NotNull FastArrayList<Move> legalMoves = game.moves(context).moves();

        // Pre-emptively obtain a legal random move.
        final int numLegalMoves = legalMoves.size();
        final int randomLegalMove = ThreadLocalRandom.current().nextInt(numLegalMoves);

        if (randomLegalMove >= numLegalMoves)
            throw new AssertionError("Error: random legal move " + randomLegalMove +
                    " is greater than the total number of possible legal moves " + numLegalMoves);

        // Determine the board's hash code to create a unique identifier.
        final int boardHashCode = Utils.boardToHashcode(context);

        // The final move we choose to make
        final int moveChoice;

        // Perform an Epsilon-Greedy Policy for choosing a move.
        if (ThreadLocalRandom.current().nextDouble(0, 1) < this.epsilon) {
            // We will return the random move.
            moveChoice = randomLegalMove;
        } else {
            // Otherwise, we select the optimal list in memory at this state.
            // Note: If there are no optimal moves from this state yet, then we will
            // choose the random move.

            // Determine the possible Q values from this state.
            final double[] QValues = this.getQValues(boardHashCode, numLegalMoves);

            if (QValues.length == 0)
                throw new AssertionError("Error: the Q values array was empty. Aborting.");

            // Find the arg max Q value. The maximum value's index is our optimal move choice.
            moveChoice = argmax(QValues);
        }

        // Finally, return the optimal move, as defined by the policy.
        // Also store the move to the move history.
        @NotNull Move selectedMove = legalMoves.get(moveChoice);

        // Ensure that the move choice is legal
//        assert moveChoice < legalMoves.size();

        // Put the moves in backwards order to the Queue, so that it is in FILO order.
        if (this.learn) {
            int[] move = new int[3];
            move[0] = boardHashCode;
            move[1] = moveChoice;
            move[2] = numLegalMoves;
            moveHistory.add(move);
        }

        return selectedMove;
    }

    /**
     * After an episode has been played, this method will induce backward episodic reward updates.
     *
     * A deque is held in memory that stores the order of the moves taken in an episode, in reverse order. Then the
     * history is iterated across from final move -> first move, applying the reward updates at each step.
     *
     * @param reward the reward from the final end state of a particular episode.
     */
    public void updateQBackwards(double reward) {
        // Only learn if we're supposed to.
        if (!this.learn)
            return;

        // Implement a backwards Queue that allows for the whole game to be played
        // And then, once the game is over, retroactively go down the queue to apply
        // Q updates in place. This way, a whole game is simulated at once before updates occur.
        // and are retroactively accounted for via the chain nature.

        // So, we assume that the whole game has been played, with a reward of zero assumed for all but
        // the very ending state.
        if (moveHistory == null) throw new AssertionError("Error: moveHistory was not instantiated. Aborting.");

        if (moveHistory.isEmpty()) throw new AssertionError("Error: moveHistory is empty when attempting to update the Q values. Aborting.");

//        @NotNull Iterator<int[]> queueIterator = moveHistory.iterator();
//
//        if(!queueIterator.hasNext()) throw new AssertionError("Error: moveHistory iterator returns no next nodes when initialized at the beginning. Aborting.");
        //
        // Obtain the first node in the queue (the last state of the game).
        //
        // These will be updated as the queue progresses to store the previous state, and hence are not final.
        //
        // As we are effectively stepping backwards through the queue, we begin with the most recent moves
        // and then update the move before it.
        //
        int[] currentMove = moveHistory.pop();
        if (currentMove.length != 3) throw new AssertionError("Error: expected all moves to be arrays of length 3.");
        int currentBoardHashcode = currentMove[0];
        int currentNumLegalMoves = currentMove[2];

        while (!moveHistory.isEmpty()) {
            // Decode the current node from the queue.
            final int[] previousMove = moveHistory.pop();
            if (previousMove.length != 3) throw new AssertionError("Error: expected all moves to be arrays of length 3.");
            final int previousBoardHashcode = previousMove[0];
            final int previousMoveChoice    = previousMove[1];
            final int previousNumLegalMoves = previousMove[2];

            final double[] currentQValuesArray = getQValues(currentBoardHashcode, currentNumLegalMoves);

            // Find the optimal Q value of the current step. This will be used to update the Q value
            // of the previous state.
            final double maxCurrentQValue = max(currentQValuesArray);

            //
            // Perform the Q-learning update.
            //
            double[] previousQValuesArray = getQValues(previousBoardHashcode, previousNumLegalMoves);
            final double initialQValue = previousQValuesArray[previousMoveChoice];
            final double updatedQValue = (1 - this.alpha) * initialQValue + this.alpha * (reward + this.gamma * maxCurrentQValue);

            // Update the array of previous Q Values, for memory efficiency.
            previousQValuesArray[previousMoveChoice] = updatedQValue;

            // Update Q.
            if(Q.get(previousBoardHashcode)== null)
                throw new AssertionError("Error: the previous board's hashcode is not kept in Q. Aborting.");
            Q.put(previousBoardHashcode, previousQValuesArray);

            // Set reward to be zero, as only the very final state of the game receives a potentially non-zero reward.
            reward = 0;

            // Update the current parameters.
            currentBoardHashcode = previousBoardHashcode;
            currentNumLegalMoves = previousNumLegalMoves;
        }

        // Reset the move history.
        moveHistory.clear();
    }


    /**
     * Retrieves the associated Q values for a particular board state. If the board state does not exist in memory yet,
     * then an array of 0s the length of the number of legal moves is added and returned.
     * @param boardHashCode the current state's unique hashcode.
     * @param numLegalMoves the number of legal moves at this position.
     * @return a double[] containing the Q values associated with the current board state.
     */
    public double[] getQValues(final int boardHashCode, final int numLegalMoves) {
        double[] qValues;

        if (Q == null) throw new AssertionError("Error: Q must be initialized. ");

        if (!Q.containsKey(boardHashCode)) {
            qValues = new double[numLegalMoves];
            Q.put(boardHashCode, qValues);
        } else {
            qValues = Q.get(boardHashCode);

            if (qValues == null) throw new AssertionError("Error: retrieved Q values are null, despite Q containing its key.");
        }

        return qValues;
    }

    /**
     * Returns the largest element within the array. If the array is empty, it returns
     * the largest negative value a double may contain.
     * @param array the list of doubles.
     * @return the largest element of the array.
     */
    public double max(final double[] array) {
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
    public int argmax(final double[] array) {

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

    public BiMap<Integer, double[]> getQ() {
        if (Q == null) throw new AssertionError("Error: Attempting to access Q, but Q is null.");
        return Q;
    }

    @Override
    public boolean supportsGame(final Game game) {
        return game.isAlternatingMoveGame();
    }

    @Override
    public void closeAI() {
        super.closeAI();
        Q = null;
        moveHistory = null;
    }

    public void setEpsilon(double eps) {
        this.epsilon = eps;
    }

}
