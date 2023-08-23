package MLP.dataset;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
public class DatasetLoader { //Loads the data set and put it in a List of DigitData.
    public static List<DigitData> loadDataSet(String file, String separator, int classIndex) {
        List<DigitData> images = null;
        try (BufferedReader reader = new BufferedReader(new FileReader(new File(file)))) //accesses the Data Set file.
        {
         images = new ArrayList<>();
         String line;
         while ((line = reader.readLine()) != null) { //read until no data is left.
         String[] arr = line.split(separator);
          double[] img;     
                img = new double[classIndex == -1 ? arr.length : (arr.length - 1)];//if no class label is in the data set create the data of complete length otherwise length will be -1.
                String classValue = null; 
                for (int i = 0; i < arr.length; i++) {//loop over all the data.
                    if (i == classIndex) {
                        classValue = arr[i]; //saves the class index.
                    } else {
                        double val;
        try {
          val = Double.parseDouble(arr[i]); //get the data value.
        		} catch (NumberFormatException e) {
        			val = Double.NaN; // in case of error save as 0.
                        }
              if (classIndex != -1 && i > classIndex)
              img[i - 1] = val; //save the data value.
              else
              img[i] = val; 
                    }
                }
             images.add(new DigitData(img, Integer.parseInt(classValue)));
           }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return images;
    }
}