package MLP.perceptron;
import MLP.perceptron.math.Vec;
public class Result { //The outcome of an evaluation.
    private final Vec output; //Declare the output data.
    private final Double cost; //Declare the cost.
    public Result(Vec output) { //Parameterised constructor.
        this.output = output; //Output data.
        cost = null;
    }
    public Result(Vec output, double cost) { //Parameterised Constructor.
        this.output = output; //Output data.
        this.cost = cost; //Cost for the output.
    }
    public Vec getOutput() {
        return output; //Output data of the current result.
    }
    public Double getCost() { //Cost for the current output.
        return cost;
    }
    @Override
    public String toString() {
        return "Result{" + "output=" + output +
            ", cost=" + cost +
            '}';
    }
}