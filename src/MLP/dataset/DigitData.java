package MLP.dataset;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.util.Random;
import static java.awt.geom.AffineTransform.getTranslateInstance;
import static java.lang.Math.*;
   public class DigitData { //Holds the data for a digit as well as its label.
    private static final double[][] EXPECTED_ARRAY = new double[][]{//Expected output for all classes.
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };
    private double[] data; //array for storing the data.
    private double[] transformedData; // array for storing the string data after transformation.
    private int label; 
    private Random rnd = new Random(); //random number generator for data.
    public DigitData(double[] data, int label) {
        this.data = data;
        this.label = label;
    }
    public double[] getData() {
        return transformedData != null ? transformedData : data; // Returns stored data, prefers transformed data if its present.
    }
    public int getLabel() {
        return label; // returns the class label.
    }
    public double[] getLabelAsArray() {
        return EXPECTED_ARRAY[label]; // returns the expected output for given class.
    }
    public void transformDigit() { // Creates a slightly modified version of the original digit.
        try {
            double[] dst = new double[data.length];
            boolean potentialOverspill;
            int overspillCounter = 0;
            do {
                potentialOverspill = false;
                AffineTransform t = getTranslateInstance(14, 14);//translate and rotate the digit data.
                t.rotate(toRadians(rnd() * 20));
                t.scale(rnd() * 0.25 + 1, rnd() * 0.25 + 1);
                t.translate(-14 + (rnd() * 3), -14 + (rnd() * 3));
                Point2D wPoint = new Point2D.Double();
                Point2D rPoint = new Point2D.Double();
                looping:
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        wPoint.setLocation(x, y);
                        t.inverseTransform(wPoint, rPoint);
                        clamp(rPoint, 0, 8);                     
                        int xi = (int) rPoint.getX();//integer part.
                        int yi = (int) rPoint.getY();                      
                        double xf = rPoint.getX() - xi; //fractional part.
                        double yf = rPoint.getY() - yi;                  
                        double interpolatedValue =      //get interpolated value .
                                (1 - xf) * (1 - yf) * pixelValue(xi, yi, data) +
                                (1 - xf) * yf * pixelValue(xi, yi + 1, data) +
                                xf * (1 - yf) * pixelValue(xi + 1, yi, data) +
                                xf * yf * pixelValue(xi + 1, yi + 1, data);
                        if (interpolatedValue > 0 && onBorder(x, y)) {
                            potentialOverspill = true;
                            overspillCounter++;
                            break looping;
                        }
                        dst[y * 8 + x] = interpolatedValue;
                    }
                }
            } while (potentialOverspill && overspillCounter < 5);
            if (overspillCounter < 5)
                transformedData = dst;
        } catch (NoninvertibleTransformException e) {
            throw new RuntimeException("Should not happen: ", e);
        }
    }

    private boolean onBorder(int x, int y) { //Utility function to verify if the pixel is on the border or not, with parameters being x and y coordinates.
        return x == 0 || y == 0 || x == 7 || y == 7; //returns true if the pixel is on border.
    }
    private double pixelValue(int x, int y, double[] data) { //returns a pixel value from the array.
        return data[min(y * 8 + x, data.length - 1)];
    }
    private void clamp(Point2D point, int min, int max) { //Clamps a given number to maximum and minimum range.
        point.setLocation(min(max(point.getX(), min), max), min(max(point.getY(), min), max));
    }
    private double rnd() {
        return rnd.nextDouble() - 0.5; //returns a random number
    }
    public void setRandom(Random rnd) { //sets a random value generator.
        this.rnd = rnd;
    }
}
