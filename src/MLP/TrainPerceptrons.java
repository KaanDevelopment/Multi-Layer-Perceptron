package MLP;
import static MLP.perceptron.Activation.Leaky_ReLU;
import static MLP.perceptron.Activation.Softmax;
import static MLP.perceptron.math.SharedRnd.getRnd;
import static MLP.perceptron.math.SharedRnd.setRnd;
import static java.lang.String.format;
import static java.lang.System.currentTimeMillis;
import static java.util.Collections.shuffle;
import static java.util.Collections.unmodifiableList;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import MLP.dataset.DatasetLoader;
import MLP.dataset.DigitData;
import MLP.perceptron.CostFunction;
import MLP.perceptron.Initializer;
import MLP.perceptron.Layer;
import MLP.perceptron.MultiLayerPerceptron;
import MLP.perceptron.Optimizer;
import MLP.perceptron.Result;
import MLP.perceptron.math.Vec;
public class TrainPerceptrons {
    private static int BATCH_SIZE = 32; //used to divide complete data sets into small batches for training.
    public static void main(String[] args) throws IOException {    
    	boolean use2Fold = false; 	//for selecting normal run mode or 2 fold mode [false will run the perceptron normally and true will perform a two fold test].
    	int seed = 942457; // the starting point for random numbers.
        setRnd(new Random(seed));     
        List<DigitData> trainData = DatasetLoader.loadDataSet("cw2DataSet1.csv",",",64);  //load the training data set.   
        List<DigitData> testData = DatasetLoader.loadDataSet("cw2DataSet2.csv",",",64); //load the test data set.   
        for (DigitData d : trainData) {
            d.setRandom(new Random(seed++));//set a random seed.
        }
        if(use2Fold) { //evaluate the Multi-layer Perceptron for 2 fold testing.       	
        	double[] fold1accuracy = trainFold(1, trainData, testData);
            double[] fold2accuracy = trainFold(2, testData, trainData);
            System.out.println(format("\nMulti-layer Perceptron Final Training Accuracy= %6.3f", 100*max(fold1accuracy[0],fold2accuracy[0])));
            System.out.println(format("\nMulti-layer Perceptron Final Testing Accuracy= %6.3f", 100*max(fold1accuracy[1],fold2accuracy[1])));   
        									} else {
        	trainFold(0, trainData, testData); 	//evaluates the Multi-layer Perceptron for normal testing.
        	}
          }
    	private static double max(double n1, double n2) { //selects the max value between n1 and n2.
    return n1 > n2? n1:n2;
    }
    private static double[] trainFold(int fold, List<DigitData> trainData, List<DigitData> testData) { //Train Multi-layer perceptron on given train set until there is no improvement in loss in the test set.
    	if(fold>0)System.out.println("\n\nTraining Multi-layer Perceptron for Fold "+fold);  	
    	MultiLayerPerceptron perceptron = //create the multi layer perceptron.
                new MultiLayerPerceptron.Builder(64)
                        .addLayer(new Layer(38, Leaky_ReLU))  //Adds first layer. 
                        .addLayer(new Layer(12, Leaky_ReLU)) //Adds second layer.
                        .addLayer(new Layer(10, Softmax))   //Adds third layer.
                        .initWeights(new Initializer.XavierNormal()) //Initialises the weights in the perceptron.
                        .setCostFunction(new CostFunction.Quadratic()) //Sets cost function for back propagation.
                        .setOptimizer(new Optimizer.GradientDescent(0.05)).create(); //sets optimiser for back propagation and creates the perceptron.
        int epoch = 0; //set the epoch (number of training runs/iterations) to 0.
        double errorRateOnTrainDS;//create a type to store train error.
        double errorRateOnTestDS;//create a type to store test error.
        StopEvaluator evaluator = new StopEvaluator(perceptron, 40, null);  //create instance of evaluator.
        boolean shouldStop = false;
        long t0 = currentTimeMillis(); //train until no progress is possible.
        do {
            epoch++;
            shuffle(trainData, getRnd());//trains the data set.      
            int correctTrainDS = applyDataToNet(trainData, perceptron, true);
            errorRateOnTrainDS = 100 - (100.0 * correctTrainDS / trainData.size()); 
            if (epoch % 5 == 0) {//test Multi-layer Perceptron performance but do not update weights.
                int correctOnTestDS = applyDataToNet(testData, perceptron, false);
                errorRateOnTestDS = 100 - (100.0 * correctOnTestDS / testData.size());
                shouldStop = evaluator.stop(errorRateOnTestDS);
                double epocsPerMinute = epoch * 60000.0 / (currentTimeMillis() - t0);
                System.out.println(format("Epoch: %3d    |   Train error rate: %6.3f %%    |   Test error rate: %5.2f %%   |   Epocs/min: %5.2f", epoch, errorRateOnTrainDS, errorRateOnTestDS, epocsPerMinute));
            }else {
            	double epocsPerMinute = epoch * 60000.0 / (currentTimeMillis() - t0);
            	System.out.println(format("Epoch: %3d    |   Train error rate: %6.3f %%    |   Epocs/min: %5.2f", epoch, errorRateOnTrainDS, epocsPerMinute));
            }
            trainData.parallelStream().forEach(DigitData::transformDigit);
        } while (!shouldStop);       
        double []accuracy = new double[] {0,0}; //calculates the confusion matrix for the training data set.
        accuracy[0] = computeConfusionMatrix("Training", trainData, perceptron);  
        accuracy[1] = computeConfusionMatrix("Testing", testData, perceptron);//calculates the confusion matrix for the testing data set.
        double lowestErrorRate = evaluator.getLowestErrorRate();
        System.out.println(format("No improvement. Reached a lowest error rate of %7.4f %%", lowestErrorRate));    	
    	return accuracy;
    }
   private static int applyDataToNet(List<DigitData> data, MultiLayerPerceptron perceptron, boolean learn) {//Run the data set through the multi-layer perceptron. If "learn" is true the perceptron will learn from the data.
        final AtomicInteger correct = new AtomicInteger(); //creates a variable that can be read and written atomically
        for (int i = 0; i <= data.size() / BATCH_SIZE; i++) {
            getBatch(i, data).parallelStream().forEach(img -> {
                Vec input = new Vec(img.getData());
                Result result = learn ?
                        perceptron.evaluate(input, new Vec(img.getLabelAsArray())) :
                        perceptron.evaluate(input);
                if (result.getOutput().indexOfLargestElement() == img.getLabel())
                    correct.incrementAndGet();
            });
            if (learn)
                perceptron.updateFromLearning();
        }
        return correct.get();
    }
    private static List<DigitData> getBatch(int i, List<DigitData> data) { //Cuts out batch "i" from data set.
        int fromIx = i * BATCH_SIZE;
        int toIx = Math.min(data.size(), (i + 1) * BATCH_SIZE);
        return unmodifiableList(data.subList(fromIx, toIx));
    }
    private static double accuracy =0;
 
    private static double  computeConfusionMatrix(String title, List<DigitData> dataset, MultiLayerPerceptron perceptron){ //calculate confusion matrix for given multi-layer perceptron.
        double[][] matrix = new double[][]{{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,0}};
        dataset.forEach(dataPoint->{
            int actual = dataPoint.getLabel();
            int predicted = getPredictedClass(perceptron, new Vec(dataPoint.getData()));
            matrix[actual][predicted] ++;
            if(actual==predicted) accuracy++;
        });accuracy/=dataset.size();
        System.out.println("\n\t\t\t\t"+title+" Results");
        System.out.println(format("Multi-layer Perceptron Accuracy: %3f",accuracy*100));
        System.out.println("\nConfusion Matrix");
        for (double[] cmDatum : matrix) {
            System.out.println(Arrays.toString(cmDatum));
        }
        double[] pa = computePrecision(matrix);
        double[] ra = computeRecall(matrix);
        System.out.println("Class Index \tPrecision \t\tRecall");
        for (int i = 0; i < pa.length; i++) {
            System.out.println(format("%d\t\t%3f\t\t%3f",i, pa[i] ,ra[i]));
        }
        return accuracy;
    }
    private static int getPredictedClass(MultiLayerPerceptron perceptron,Vec data ) { //Get prediction output for given perceptron and returns the class index of prediction. 
        return perceptron.evaluate(data).getOutput().indexOfLargestElement();
    }
    private static double[] computePrecision(double[][] matrix){ //Get prediction output for the given perceptron and then calculates class wise precision from the given confusion matrix, 
        double[] precisionArray = new double[]{0,0,0,0,0,0,0,0,0,0}; // and then returns class wise precision.
        for (int i = 0; i < matrix.length; i++) {
            double sum =0;
            for (int i1 = 0; i1 < matrix.length; i1++) sum +=matrix[i][i1];
            precisionArray[i] = 100*(matrix[i][i])/(sum); //Precision = TP/(TP+FN).
       }
        return precisionArray;
    }
    private static double[] computeRecall(double[][] matrix){ //Calculates class wise recall form the given confusion matrix and returns the class wise recall.
    	double[] recallArray = new double[]{0,0,0,0,0,0,0,0,0,0};
    	for (int i = 0; i < matrix.length; i++) {
    	double sum =0;
    	for (int i1 = 0; i1 < matrix.length; i1++) sum +=matrix[i1][i];
    	recallArray[i] = 100*(matrix[i][i])/(sum);
    	}
    	return recallArray;
    	}
}