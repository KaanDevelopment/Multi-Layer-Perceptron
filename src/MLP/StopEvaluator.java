package MLP;
import java.util.LinkedList;
import MLP.perceptron.MultiLayerPerceptron;
class StopEvaluator { //The StopEvaluator keeps track of whether it is meaningful to continue to train the Perceptron or if the error rate of the test data seems to be on the rise(if the perceptron starts over fitting).
    private int windowSize; 
    private Double acceptableErrorRate;
    private final LinkedList<Double> errorRates;
    private double lowestErrorRate = Double.MAX_VALUE;
    private double lastErrorAverage = Double.MAX_VALUE;
    public StopEvaluator(MultiLayerPerceptron network, int windowSize, Double acceptableErrorRate) {
        this.windowSize = windowSize;
        this.acceptableErrorRate = acceptableErrorRate;
        this.errorRates = new LinkedList<>();
    }
    public boolean stop(double errorRate) {  //See if there is any point in continuing ...
        if (errorRate < lowestErrorRate) {
            lowestErrorRate = errorRate;
        }
        if (acceptableErrorRate != null && lowestErrorRate < acceptableErrorRate) return true;      
        errorRates.addLast(errorRate);  //update moving average.
        if (errorRates.size() < windowSize) { 
            return false;   // never stop if we have not filled moving average.
        }
        if (errorRates.size() > windowSize)
            errorRates.removeFirst();
        double avg = getAverage(errorRates);
        if (avg > lastErrorAverage) { 
            return true;
        } else { // see if we should stop.
            lastErrorAverage = avg;
            return false;
        }
    } 
    public double getLowestErrorRate() {
        return lowestErrorRate; // returns the lowest error rate.
    }
    private double getAverage(LinkedList<Double> list) { // returns the average of a inputed list.
        return list.stream().mapToDouble(Double::doubleValue).average().getAsDouble();
    }
}