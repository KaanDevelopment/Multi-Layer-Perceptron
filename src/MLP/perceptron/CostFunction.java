package MLP.perceptron;
import MLP.perceptron.math.Vec;
public interface CostFunction { //This class uses a Quadratic cost function used for back propagation.
    String getName(); //returns the name of cost function.
    double getTotal(Vec expected, Vec actual); //Calculates the error rate for prediction.
    Vec getDerivative(Vec expected, Vec actual); //the expected prediction of vector, the actual predicted value gets the derivative for prediction.
    class Quadratic implements CostFunction {
        @Override
        public String getName() { //Cost function: Quadratic, C = ∑(y−exp)^2
            return "Quadratic";
        }
        @Override
        public double getTotal(Vec expected, Vec actual) {
            Vec diff = actual.sub(expected); //the expected prediction.
            return diff.dot(diff); //the actual predicted value.
        }
        @Override
        public Vec getDerivative(Vec expected, Vec actual) { //Calculates the derivative of the prediction for gradient descent.
            return actual.sub(expected).mul(2); //Returns the cost for the prediction.
        }
    }
}
