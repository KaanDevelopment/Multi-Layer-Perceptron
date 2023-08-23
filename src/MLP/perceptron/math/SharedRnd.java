package MLP.perceptron.math;
import java.util.Random;
public class SharedRnd {
    private static Random rnd = new Random(); //Global random initialiser for perceptron
    public static Random getRnd() { // random number generator
        return rnd;
    }
    public static void setRnd(Random rnd) {
        SharedRnd.rnd = rnd; // sets the random number generator
    }
}
