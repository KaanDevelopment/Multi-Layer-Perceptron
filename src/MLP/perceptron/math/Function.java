package MLP.perceptron.math;
@FunctionalInterface
public interface Function { // helps to accommodate different functions for activation and operation mapping.
    double apply(double value);
}
