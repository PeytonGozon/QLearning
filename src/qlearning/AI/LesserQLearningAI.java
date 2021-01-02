package qlearning.AI;

import game.Game;
import main.collections.FastArrayList;
import org.jetbrains.annotations.NotNull;
import util.AI;
import util.Context;
import util.Move;
import utils.AIUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;

public class LesserQLearningAI extends AI {

    // Player ID
    protected int player = -1;

    // Learning rate, discount rate, epsilon-greedy policy parameter.
    private final double alpha, gamma, epsilon;

    // The underlying Q table of form Q[Legal Moves] = [Value of Each Move];
    public HashMap<FastArrayList<Move>, Double[]> Q = null;
    private FastArrayList<Move> legalMovesLastTurn = null;
    private int lastMoveMade = -1;

    /**
     * Constructor
     * @param alpha the learning rate.
     * @param gamma the future reward discount rate.
     * @param epsilon the parameter for the epsilon-greedy policy parameter
     */
    public LesserQLearningAI(double alpha, double gamma, double epsilon) {
        this.friendlyName = "Q-Learning Agent (legal move -> value)";
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
    }

    @Override
    public void initAI(final Game game, final int playerID) {
        this.player = playerID;

        this.Q = new HashMap<>();
    }

    @Override
    public Move selectAction(final Game game,
                             final Context context,
                             final double maxSeconds,
                             final int maxIterations,
                             final int maxDepth) {
        // Deep copy the allowed moves.
        final FastArrayList<Move> allowedMoves = getLegalMoves(game, context);
        this.legalMovesLastTurn = new FastArrayList<>();
        for (Move m : allowedMoves)
            this.legalMovesLastTurn.add(m);

        // Obtain a random move ahead of time, should the optimal Q value not be fully explored yet.
        this.lastMoveMade = ThreadLocalRandom.current().nextInt(this.legalMovesLastTurn.size());

        // Perform an Epsilon-Greedy Policy for choosing a move
        // Otherwise, select the optimal move from the list in memory at this state.
        if (ThreadLocalRandom.current().nextDouble(0, 1) >= 1 - this.epsilon) {
            // The LesserQLearningAI defines the "State" to instead be defined as the set of all legal moves
            // the AI can currently make.
            final Double[] QValues = getQValues(this.legalMovesLastTurn);

            // Perform an arg-max across the current Q values.
            double maxQValue = 0.0;
            int maxQIndex = 0;

            for(int i = 0; i < QValues.length; i++)
                if (maxQValue < QValues[i]) {
                    maxQIndex = i;
                    maxQValue = QValues[i];
                }

            // If the optimal value is greater than 0, play it. Otherwise, play the random move from above.
            if (maxQValue > 0)
                this.lastMoveMade = maxQIndex;

        }

        return this.legalMovesLastTurn.get(this.lastMoveMade);
    }

    public void updateQ(final Game game, final Context context, final float reward) {
//        System.out.println("\tUpdating Agent " + this.player + " with a reward of " + reward);
        // Obtain the set of moves from last round and ensure it is non null.
        final FastArrayList<Move> lastRoundLegalMoves = this.legalMovesLastTurn;

        if (lastRoundLegalMoves == null) {
            System.out.println("Error: The set of moves made last round is NULL! Last move index: " + this.lastMoveMade);
            return;
        }

        // Obtain the set of legal moves this round and ensure it is non null.

        final FastArrayList<Move> currentRoundLegalMoves = getLegalMoves(game, context);

        if (currentRoundLegalMoves == null) {
            System.out.println("Error: The current set of legal moves is NULL!");
            return;
        }

        // Obtain the maximum Q value associated with the set of legal moves.
        Double currentRoundBestQValue = 0.0;

        // Obtain the maximum Q value associated with this current state.
        if (!currentRoundLegalMoves.isEmpty()) {
            Double[] currentRoundQValues = getQValues(currentRoundLegalMoves);
            for (Double d : currentRoundQValues)
                if (d < currentRoundBestQValue)
                    currentRoundBestQValue = d;
        }

        // Perform the Q-Learning Update to the previous Q-value.
        Double[] lastRoundQValues = getQValues(lastRoundLegalMoves);

        // Ensure we have a valid move index.
        if (this.lastMoveMade >= 0 && this.lastMoveMade < lastRoundQValues.length) {
            double oldQValue = lastRoundQValues[this.lastMoveMade];
            double newQValue = oldQValue + this.alpha * (reward + this.gamma * currentRoundBestQValue - oldQValue);

            lastRoundQValues[this.lastMoveMade] = newQValue;

            // Update the hashmap.
            this.Q.put(lastRoundLegalMoves, lastRoundQValues);
        }

        Utils.boardToHashcode(context);
    }

    /**
     * Get the Q values associated with a particular state.
     * @param legalMoves FastArrayList<Move> containing all legal moves.
     * @return Double[] containing the Q-value associated with each move.
     */
    public Double[] getQValues(@NotNull FastArrayList<Move> legalMoves) {
        if (!this.Q.containsKey(legalMoves)) {
            Double[] moveValues = new Double[legalMoves.size()];
            Arrays.fill(moveValues, 0.0);
            this.Q.put(legalMoves, moveValues);
        }
        return this.Q.get(legalMoves);
    }

    /**
     * Get the legal moves for this player currently.
     * @param game The game object.
     * @param context The current context.
     * @return FastArrayList<Move> containing all legal moves.
     */
    public FastArrayList<Move> getLegalMoves(final @NotNull Game game, final Context context) {
        // Obtain the set of legal moves from the current game state and board.
        FastArrayList<Move> legalMoves = game.moves(context).moves();

        // Allow for simultaneous in-game moves, which may change the valid set of moves for this AI.
        if (!game.isAlternatingMoveGame())
            legalMoves = AIUtils.extractMovesForMover(legalMoves, player);

//        System.out.println("\tThe set of legal moves are:\n\t\t"+legalMoves+"\n");

        return legalMoves;
    }
}
