package MLP.perceptron;
import MLP.perceptron.math.Matrix;
import MLP.perceptron.math.Vec;
public interface Optimizer {
    void updateWeights(Matrix weights, Matrix dCdW);
    Vec updateBias(Vec bias, Vec dCdB);
    Optimizer copy();
    class GradientDescent implements Optimizer { //Updates the Weights and biases based on a constant learning rate - i.e. W -= η * dC/dW
        double learningRate;
        public GradientDescent(double learningRate){
            this.learningRate=learningRate;
        }
        @Override
        public void updateWeights(Matrix weights, Matrix dCdW) { //update The Multi-layer Perceptron weights.
            weights.sub(dCdW.mul(learningRate));
        }
        @Override
        public Vec updateBias(Vec bias, Vec dCdB) { //update The Multi-layer Perceptron bias.
            return bias.sub(dCdB.mul(learningRate));
        }
        public Optimizer copy() { //creates a copy of the optimiser.
           //  no need to make copies since this optimiser has
          //   no state. Same instance can be used by all layers.
            return this;
    
        }
    }
}
