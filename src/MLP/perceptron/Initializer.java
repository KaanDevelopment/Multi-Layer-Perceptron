package MLP.perceptron;
import static MLP.perceptron.math.SharedRnd.getRnd;
import MLP.perceptron.math.Matrix;
public interface Initializer { //This class is used to setup initial weights for the perceptron. using weight tensors and distribution equations to calculate weights.
    void initWeights(Matrix weights, int layer); //Initialises the weights for the given layer, and the given the parameters of the layer index and the weights.
    class Random implements Initializer { //Initialiser that generates tensors with a normal distribution.
        double min;
        double max;
        public Random(double min, double max){
            this.min = min;
            this.max = max;
        }
        @Override
        public void initWeights(Matrix weights, int layer) { //Initialises the weights.
            double delta = max - min;
            weights.map(value -> min + getRnd().nextDouble() * delta);
        }
    }
    class XavierUniform implements Initializer { //This draws samples from a uniform distribution within a negative limit, where limit is square root(6 / (fan_in + fan_out))
        @Override //and  fan_in is the number of input units in the weight tensor and fan_out is the number of output units.
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(6.0 / (weights.cols() + weights.rows()));
            weights.map(value -> (getRnd().nextDouble() - 0.5) * factor);
        }
    }
    class XavierNormal implements Initializer { //This draws samples from a truncated normal distribution centred on 0  * with standard deviation = square root(2 / (fan_in + fan_out))
        @Override // and fan_in is the number of input units in the weight tensor and fan_out is the number of output units.
        public void initWeights(Matrix weights, int layer) {
            final double factor = Math.sqrt(2.0 / (weights.cols() + weights.rows()));
            weights.map(value -> getRnd().nextGaussian() * factor);
        }
    }
    class LeCunUniform implements Initializer { //This draws samples from a uniform distribution within -limit, where limit is square root(3 / fan_in) and fan_in is the number of input units in the weight tensor.
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(3.0 / weights.cols());
            weights.map(value -> (getRnd().nextDouble() - 0.5) * factor);
        }
    }
    class LeCunNormal implements Initializer { //This draws samples from a truncated normal distribution centred on 0 with standard deviation <- square root(1 / fan_in) and fan_in is the number of input units.
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 1.0 / Math.sqrt(weights.cols());
            weights.map(value -> getRnd().nextGaussian() * factor);
        }
    }
    class HeUniform implements Initializer { //This draws samples from a uniform distribution within -limit, and where limit is square root(6 / fan_in) and fan_in is the number of input units in the weight tensor.
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(6.0 / weights.cols());
            weights.map(value -> (getRnd().nextDouble() - 0.5) * factor);
        }
    }
    class HeNormal implements Initializer { //This draws samples from a truncated normal distribution centred on 0 with standard deviation = square root(2 / fan_in) and where fan_in is the number of input units.
        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = Math.sqrt(2.0 / weights.cols());
            weights.map(value -> getRnd().nextGaussian() * factor);
        }
    }
}
