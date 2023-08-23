package MLP.perceptron;
import java.util.ArrayList;
import java.util.List;
import MLP.perceptron.math.Matrix;
import MLP.perceptron.math.Vec;
public class MultiLayerPerceptron {
    private final CostFunction costFunction;//declare the Cost function for Multi-layer Perceptron.
    private final int perceptronInputSize; //declare the perceptron input shape.
    private final double l2;// declare the L2 factor for the Multi-layer Perceptron (acts like a force that removes a small percentage of each weights at each iterations).
    private final Optimizer optimizer; //declare Optimiser for the Multi-layer Perceptron.
    private final List<Layer> layers = new ArrayList<>(); //List for all layers in the Multi-layer Perceptron.
    private MultiLayerPerceptron(Builder nb) { //Creates a Multi-layer Perceptron given the configuration set in the builder.
        costFunction = nb.costFunction; //where "nb" is the parameter to configure the Multi-layer Perceptron.
        perceptronInputSize = nb.perceptronInputSize;
        optimizer = nb.optimizer;
        l2 = nb.l2;
        Layer inputLayer = new Layer(perceptronInputSize, Activation.Identity); 
        layers.add(inputLayer); //Adding a input Layer.
        Layer precedingLayer = inputLayer;
        for (int i = 0; i < nb.layers.size(); i++) {
            Layer layer = nb.layers.get(i);
            Matrix w = new Matrix(precedingLayer.size(), layer.size());
            nb.initializer.initWeights(w, i);
            layer.setWeights(w); //Each layer contains the weights between preceding layer and itself.
            layer.setOptimizer(optimizer.copy());
            layer.setL2(l2);
            layer.setPrecedingLayer(precedingLayer);
            layers.add(layer);
            precedingLayer = layer;
        }
    }  
    public Result evaluate(Vec input) { //Evaluates an input vector, returning the Perceptrons' output, without cost or learning anything from it.
        return evaluate(input, null);
    } 
    public Result evaluate(Vec input, Vec expected) { //Evaluates an input vector, returning the Perceptrons' output.  If the parameter "expected" is specified the result will contain 
        Vec signal = input; 						 //a cost and the perceptron will gather some learning from this operation.
        for (Layer layer : layers)
            signal = layer.evaluate(signal);
        if (expected != null) {
            learnFrom(expected);
            double cost = costFunction.getTotal(expected, signal);
            return new Result(signal, cost);
        }
        return new Result(signal);
    }
    private void learnFrom(Vec expected) { //This will gather some learning based on the expected vector and how that differs to the actual output from the perceptron.
        Layer layer = getLastLayer(); 	  //This difference (or error) is backpropagated through the net. To make it possible to use, small batches the learning is not immediately realized.
        Vec dCdO = costFunction.getDerivative(expected, layer.getOut());//The error is initially the derivative of the cost-function. 
        do { //iterate backwards through the layers
            Vec dCdI = layer.getActivation().dCdI(layer.getOut(), dCdO);
            Matrix dCdW = dCdI.outerProduct(layer.getPrecedingLayer().getOut());        
            layer.addDeltaWeightsAndBiases(dCdW, dCdI);     //Store the deltas for weights and biases.
            dCdO = layer.getWeights().multiply(dCdI);   //prepare error propagation and store for next iteration.
            layer = layer.getPrecedingLayer();
        }
        while (layer.hasPrecedingLayer());  //Stop when we are at input layer.
    }
    public synchronized void updateFromLearning() { // Let all the gathered (but not yet realised) learning "sink in".
        for (Layer l : layers) // That is: Update the weights and biases based on the deltas collected during evaluation & training.
            if (l.hasPrecedingLayer())         //Skip the input layer.
                l.updateWeightsAndBias();
    }
    public List<Layer> getLayers() {
        return layers; //returns all layers in the Multi-layer Perceptron.
    }
    private Layer getLastLayer() { //returns the final classification layer of perceptron.
        return layers.get(layers.size() - 1);
    }
    public static class Builder { //builder for the perceptron.
        private final List<Layer> layers = new ArrayList<>();
        private final int perceptronInputSize;
        private Initializer initializer = new Initializer.Random(-0.5, 0.5);
        private CostFunction costFunction = new CostFunction.Quadratic();
        private Optimizer optimizer = new Optimizer.GradientDescent(0.005);
        private double l2 = 0;
        public Builder(int perceptronInputSize) {
            this.perceptronInputSize = perceptronInputSize;
        }
        public Builder(MultiLayerPerceptron other) { //Create a builder from an existing Multi-layer Perceptron, hence making it possible to do a copy of the entire state and modify as needed.
            perceptronInputSize = other.perceptronInputSize;
            costFunction = other.costFunction;
            optimizer = other.optimizer;
            l2 = other.l2;
            List<Layer> otherLayers = other.getLayers();
            for (int i = 1; i < otherLayers.size(); i++) {
                Layer otherLayer = otherLayers.get(i);
                layers.add(new Layer(otherLayer.size(),otherLayer.getActivation(),otherLayer.getBias()));
            }
           initializer = (weights, layer) -> {
                Layer otherLayer = otherLayers.get(layer + 1);
                Matrix otherLayerWeights = otherLayer.getWeights();
                weights.fillFrom(otherLayerWeights);
            };
        }
        public Builder initWeights(Initializer initializer) {
            this.initializer = initializer; //Initialise perceptron layer weights.
            return this;
        }
        public Builder setCostFunction(CostFunction costFunction) { // Sets the perceptron cost function.
            this.costFunction = costFunction;
            return this;
        }
        public Builder setOptimizer(Optimizer optimizer) { //sets the perceptron optimiser.
            this.optimizer = optimizer;
            return this;
        }
        public Builder l2(double l2) { //sets the Perceptrons' l2 regulization.
            this.l2 = l2;
            return this;
        }
        public Builder addLayer(Layer layer) { //add in the layers to the Multi-layer Perceptron.
            layers.add(layer);
            return this;
        }
        public MultiLayerPerceptron create() { //Creates a new Multi-layer Perceptron according to the given parameter.
            return new MultiLayerPerceptron(this);
        }
    }
    public static class PerceptronState { //Utility class for storing and accessing the perceptron's current state.
        String costFunction; 
        Layer.LayerState[] layers;
        public PerceptronState(MultiLayerPerceptron perceptron) {
            costFunction = perceptron.costFunction.getName();
            layers = new Layer.LayerState[perceptron.layers.size()]; //make a copy of the Multi-layer Perceptrons layers.
            for (int l = 0; l < perceptron.layers.size(); l++) {
                layers[l] = perceptron.layers.get(l).getState();
            }
        }
        public Layer.LayerState[] getLayers() {
            return layers; //returns all layers in the Multi-layer Perceptron.
        }
    }
}