package MLP.perceptron;
import static java.lang.Math.exp;

import MLP.perceptron.math.Function;
import MLP.perceptron.math.Vec;
public class Activation { // Activation functions for Multi-layer Perceptrons' layers.
    private final String name;
    private Function fn;     //create the activation function instance.
    private Function dFn;  //create the derivative function for activation function.
    public Activation(String name) { 
        this.name = name;
    }
    public Activation(String name, Function fn, Function dFn) {
        this.name = name; //Activation name.
        this.fn = fn; //activation function instance.
        this.dFn = dFn; //derivative function for activation function.
    }
    public Vec fn(Vec in) { //Apply the activation function on the inputed data and the input Vector data on which the activation is to be applied.
        return in.map(fn); //activation result.
    }
    public Vec dFn(Vec out) { //Applies the derivative function on inputed data and the Vector data on which the derivative is to be applied.
        return out.map(dFn); //derivative result.
    }
    public Vec dCdI(Vec out, Vec dCdO) { // it is just a matter of multiplying.
        return dCdO.elementProduct(dFn(out));
    }
    public String getName() {
        return name;
    }
    public static Activation ReLU = new Activation( //Rectified  Linear Unit y = max(0, x).
        "ReLU",
        x -> x <= 0 ? 0 : x,                //the function.
        x -> x <= 0 ? 0 : 1                 //the derivative of the Function.
    );
    public static Activation Leaky_ReLU = new Activation( // Leaky Rectified linear unit is a variant of Rectified Linear Unit. Instead of being 0 when z<0, a leaky Rectified Linear Unit.
        "Leaky_ReLU",									//  allows a small, non-zero, constant gradient α (Normally, α=0.01).
        x -> x <= 0 ? 0.01 * x : x,         //the function.
        x -> x <= 0 ? 0.01 : 1              //the derivative of the Function.
    );
 
    public static Activation Sigmoid = new Activation(  //apples sigmoid(x) = 1 / (1 + exponent(-x)). in the results.
        "Sigmoid",
        Activation::sigmoidFn,                      // the function.
        x -> sigmoidFn(x) * (1.0 - sigmoidFn(x))    // the derivative of the Function.
    );
    public static Activation Identity = new Activation( // x=x. (used to return same value).
        "Identity",
        x -> x,                             //the function.
        x -> 1                              //the derivative of the Function.
    );
    public static Activation Softmax = new Activation("Softmax") { // converts output to a percentage. softmax = exponent(x) / reduce_sum(exponent(x)). 
        @Override
        public Vec fn(Vec in) {
            double[] data = in.getData();
            double sum = 0;
            double max = in.max();    //translate the input by the largest element to avoid overflow.
            for (double a : data)
                sum += exp(a - max);
            double finalSum = sum;
            return in.map(a -> exp(a - max) / finalSum);
        }
        @Override
        public Vec dCdI(Vec out, Vec dCdO) {
            double x = out.elementProduct(dCdO).sumElements();
            Vec sub = dCdO.sub(x);
            return out.elementProduct(sub);
        }
    };
    private static double sigmoidFn(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
}
