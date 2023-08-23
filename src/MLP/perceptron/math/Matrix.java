package MLP.perceptron.math;
import java.util.Arrays;
import java.util.StringJoiner;
import static java.lang.String.format;
import static java.lang.System.arraycopy;
import static java.util.Arrays.stream;
public class Matrix { //Matrix implementation for the mathematical operations for the neural network.
    private final double[][] data;
    private final int rows;
    private final int cols;
    public Matrix(double[][] data) { //The Parameterised constructor with data for the matrix.
        this.data = data;
        rows = data.length;
        cols = data[0].length;
    }
    public Matrix(int rows, int cols) { //creates an empty matrix of rows x columns shape.
        this(new double[rows][cols]);
    }
    public Vec multiply(Vec v) { //Multiplies current matrix with the inputed matrix v.
        double[] out = new double[rows];
        for (int y = 0; y < rows; y++)
            out[y] = new Vec(data[y]).dot(v);
        return new Vec(out);
    }
    public Matrix map(Function fn) { //Applies the function on matrix.
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] = fn.apply(data[y][x]);
        return this;
    }
    public int rows() { //total rows in the matrix.
        return rows;
    }
    public int cols() { //total columns in the matrix.
        return cols;
    }
    public Matrix mul(double s) { //Multiplies the matrix value by the constant factor "s".
        return map(value -> s * value);
    }
    public double[][] getData() {
        return data; // returns the matrix data.
    }
    public Matrix add(Matrix other) {
        assertCorrectDimension(other);
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] += other.data[y][x];
        return this;
    }

    public Matrix sub(Matrix other) { // Subtracts the inputed matrix from the current Matrix.
        assertCorrectDimension(other); //verifies that both matrixes have correct dimension.
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y][x] -= other.data[y][x];
        return this;
    }
    public Matrix fillFrom(Matrix other) { //Creates copy of the inputed matrix and saves it.
        assertCorrectDimension(other);
        for (int y = 0; y < rows; y++)
            if (cols >= 0) arraycopy(other.data[y], 0, data[y], 0, cols);
        return this;
    }
    public double average() { //average value of the matrix.
        return stream(data).flatMapToDouble(Arrays::stream).average().getAsDouble();
    }
    public double variance() { //variance in the matrix.
        double avg = average();
        return stream(data).flatMapToDouble(Arrays::stream).map(a -> (a - avg) * (a - avg)).average().getAsDouble();
    }
    private void assertCorrectDimension(Matrix other) { //Verify that inputed matrix have same shape as of this. 
        if (rows != other.rows || cols != other.cols)
            throw new IllegalArgumentException(format("Matrix of different dim: Input is %d x %d, Vec is %d x %d", rows, cols, other.rows, other.cols));
    }
    public Matrix copy() { //creates a deep copy of the matrix to be used for back propagation.
        Matrix m = new Matrix(rows, cols);
        for (int y = 0; y < rows; y++)
            if (cols >= 0) arraycopy(data[y], 0, m.data[y], 0, cols);
        return m;
    }
    @Override
    public String toString() { //String version of matrix.
        return new StringJoiner(", ", Matrix.class.getSimpleName() + "[", "]")
            .add("data=" + Arrays.deepToString(data))
            .toString();
    }
}