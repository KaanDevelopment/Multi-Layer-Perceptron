package MLP.perceptron.math;
import java.util.Arrays;
import java.util.stream.DoubleStream;
import static java.lang.String.format;
import static java.util.Arrays.stream;
public class Vec { //Vector implementation for the multi-layer perceptrons calculations.
    private final double[] data;
    public Vec(double... data) { //creates vector instance of double data
        this.data = data;
    }
    public Vec(int... data) { //creates vector instance from int data
        this(stream(data).asDoubleStream().toArray());
    }
    public Vec(int size) { //creates empty vector of a given size
        data = new double[size];
    }
    public int dimension() {
        return data.length; // vector shape i.e. the total number of elements in vector
    }
    public double dot(Vec u) { //calculate a dot product for the inputed vector "U"
        assertCorrectDimension(u.dimension()); //verify that both vectors have same dimensions
        double sum = 0;
        for (int i = 0; i < data.length; i++)
            sum += data[i] * u.data[i];
        return sum;
    }
    public Vec map(Function fn) { //Apply the given function on the vector
        double[] result = new double[data.length]; 
        for (int i = 0; i < data.length; i++)
            result[i] = fn.apply(data[i]);
        return new Vec(result);
    }
    public double[] getData() { //gets the data in the vector
        return data;
    }
    @Override
    public String toString() {
        return "Vec{" + "data=" + Arrays.toString(data) + '}';
    }
    public int indexOfLargestElement() { //index of element with largest value
        int ixOfLargest = 0;
        for (int i = 0; i < data.length; i++)
            if (data[i] > data[ixOfLargest]) ixOfLargest = i;
        return ixOfLargest;
    }
    public Vec sub(Vec u) { // Subtracts inputed vector "u" from current vector 
        assertCorrectDimension(u.dimension());
        double[] result = new double[u.dimension()];
        for (int i = 0; i < data.length; i++)
            result[i] = data[i] - u.data[i];
        return new Vec(result);
    }
    @Override
    public boolean equals(Object o) { //Checks if two vectors are equal or not
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Vec vec = (Vec) o;
        return Arrays.equals(data, vec.data);
    }
    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }
    public Vec mul(double s) { //Scales the vector data to give a constant
        return map(value -> s * value);
    }
    public Matrix outerProduct(Vec u) { // Calculates a cross product with the inputed vector u
        double[][] result = new double[u.dimension()][dimension()];
        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < u.data.length; j++)
                result[j][i] = data[i] * u.data[j];
        return new Matrix(result);
    }

    public Vec elementProduct(Vec u) { //  Calculates a element wise scaler product 
        assertCorrectDimension(u.dimension());
        double[] result = new double[u.dimension()];
        for (int i = 0; i < data.length; i++)
            result[i] = data[i] * u.data[i];
        return new Vec(result);
    }
    public Vec add(Vec u) {
        assertCorrectDimension(u.dimension()); //  implements "V" + "u"
        double[] result = new double[u.dimension()];
        for (int i = 0; i < data.length; i++)
            result[i] = data[i] + u.data[i];
        return new Vec(result);
    }
    public Vec mul(Matrix m) { //Multiply the matrix data with the vector
        assertCorrectDimension(m.rows());
        double[][] mData = m.getData();
        double[] result = new double[m.cols()];
        for (int col = 0; col < m.cols(); col++)
            for (int row = 0; row < m.rows(); row++)
                result[col] += mData[row][col] * data[row];
        return new Vec(result);
    }
    private void assertCorrectDimension(int inpDim) { //  makes sure that the matrix has correct dimensions for multiplication
        if (dimension() != inpDim)
            throw new IllegalArgumentException(format("Different dimensions: Input is %d, Vec is %d", inpDim, dimension()));
    } 
    public double max() { // max value in the vector
        return DoubleStream.of(data).max().getAsDouble();
    }
    public Vec sub(double a) { // Subtracts the constant value from the vector
        double[] result = new double[dimension()]; 
        for (int i = 0; i < data.length; i++)
            result[i] = data[i] - a;
        return new Vec(result);
    }
    public double sumElements() { // sum all the elements in the vector
        return DoubleStream.of(data).sum();
    }
}
