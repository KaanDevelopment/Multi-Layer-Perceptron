package MLP.perceptron;
import MLP.perceptron.math.Matrix;
import MLP.perceptron.math.Vec;
public class Layer { //A single layer in the perceptron. Contains the weights and biases coming into this layer.
    private final int size;
    private final ThreadLocal<Vec> out = new ThreadLocal<>();
    private final Activation activation;
    private Optimizer optimizer;
    private Matrix weights;
    private Vec bias;
    private double l2 = 0;
    private Layer precedingLayer;
    private transient Matrix deltaWeights;
    private transient Vec deltaBias;
    private transient int deltaWeightsAdded = 0;
    private transient int deltaBiasAdded = 0;
    public Layer(int size, Activation activation) {
        this(size, activation, 0);
    }
    public Layer(int size, Activation activation, double initialBias) { //initialise the Perceptron layer with parameters containing layer size, activation function for the layer and a layer weights initialiser.
        this.size = size;
        bias = new Vec(size).map(x -> initialBias);
        deltaBias = new Vec(size);
        this.activation = activation;
    }
    public Layer(int size, Activation activation, Vec bias) { //Initialise the Perceptron layer.
        this.size = size;
        this.bias = bias;
        deltaBias = new Vec(size);
        this.activation = activation;
    }
    public int size() {
        return size; //returns layer size.
    }
    public Vec evaluate(Vec i) { //Feed the input vector, "i", through this layer. Stores a copy of the output vector.
        if (!hasPrecedingLayer()) {
            out.set(i);    //No calculation "i" required for the input layer, just store the data.
        } else {
            out.set(activation.fn(i.mul(weights).add(bias))); // returns the output vector o (i.e. the result of o = iW + b).
        }
        return out.get();
    }
    public Vec getOut() { 
        return out.get(); //returns the layers output.
    }
    public Activation getActivation() {
        return activation; //returns the layers activation function.
    }
    public void setWeights(Matrix weights) { //set the weights for the layer.
        this.weights = weights;
        deltaWeights = new Matrix(weights.rows(), weights.cols());
    }
    public void setOptimizer(Optimizer optimizer) { //optimiser function for layer.
        this.optimizer = optimizer;
    } 
    public void setL2(double l2) { //l2 regulization for layer.
        this.l2 = l2;
    } 
    public Matrix getWeights() {
        return weights; //returns the layer weights.
    }
    public Layer getPrecedingLayer() {
        return precedingLayer; //returns the Perceptrons' layer next to this one.
    }

    public void setPrecedingLayer(Layer precedingLayer) { //The Multi-layer Perceptrons' next layer.
        this.precedingLayer = precedingLayer;
    }
    public boolean hasPrecedingLayer() { //returns true if the Perceptron has any next layer.
        return precedingLayer != null;
    } 
    public Vec getBias() {
        return bias; 
    }
    public synchronized void addDeltaWeightsAndBiases(Matrix dW, Vec dB) { //Add upcoming changes to the Weights and Biases.[This does not mean that the perceptron is updated].
        deltaWeights.add(dW);
        deltaWeightsAdded++;
        deltaBias = deltaBias.add(dB);
        deltaBiasAdded++;
    }
    public synchronized void updateWeightsAndBias() { //Takes an average of all added Weights and Biases and tells the optimiser to apply them to the current weights and biases.
        if (deltaWeightsAdded > 0) {
            if (l2 > 0) //Also applies L2 regulization on the weights if used.
                weights.map(value -> value - l2 * value);
            Matrix average_dW = deltaWeights.mul(1.0 / deltaWeightsAdded);
            optimizer.updateWeights(weights, average_dW);
            deltaWeights.map(a -> 0);   
            deltaWeightsAdded = 0;
        		}
        		if (deltaBiasAdded > 0) {
        			Vec average_bias = deltaBias.mul(1.0 / deltaBiasAdded);
        			bias = optimizer.updateBias(bias, average_bias);
        			deltaBias = deltaBias.map(a -> 0);
        			deltaBiasAdded = 0;
        		}
    	}
    public LayerState getState() { //returns reference to current layer.
        return new LayerState(this);
    }
    public static class LayerState { //Utility class for storing and accessing the layers data.
        double[][] weights;
        double[] bias;
        String activation;
        public LayerState(Layer layer) {
            weights = layer.getWeights() != null ? layer.getWeights().getData() : null;
            bias = layer.getBias().getData();
            activation = layer.activation.getName();
        }
        public double[][] getWeights() {
            return weights; //returns the layer weights.
        }
        public double[] getBias() {
            return bias; // returns the layer bias.
        }
    }
}
