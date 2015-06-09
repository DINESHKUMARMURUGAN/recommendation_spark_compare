

import org.apache.spark.*;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
public class SparkMapReduce {
    
	public static void main(String[] args) {
	JavaSparkContext sc = new JavaSparkContext("local[2]", "First Spark App");
    // we take the raw data in CSV format and convert it into a set of records of the form (user, product, price)
    JavaRDD<String[]> data = sc.textFile("hdfs://127.0.0.1/user/cloudera/dineshinput/pml-training.csv")
    .map(new Function<String, String[]>() {
      @Override
      public String[] call(String s) throws Exception {
        return s.split(",");
      }
    });
    
    
    JavaRDD<String> datauser = sc.textFile("hdfs://127.0.0.1/user/cloudera/dineshinput/u.data");
    		
    		
    JavaRDD<Rating> ratings = datauser.map(new Function<String, Rating>() {
    	      @Override
    	      public Rating call(String s) throws Exception {
    	        String[] sarray =  s.split("\\t");
    	        return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]), 
                        Double.parseDouble(sarray[2]));
    	      }
    	    });
    
 
    int rank = 10;
    int numIterations = 20;
    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01); 
    
    
    JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
    	      new Function<Rating, Tuple2<Object, Object>>() {
    	        public Tuple2<Object, Object> call(Rating r) {
    	          return new Tuple2<Object, Object>(r.user(), r.product());
    	        }
    	      }
    	    );
    
    System.out.println("#########---UserProduct---##########"+userProducts.first());
    
    JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
    	      model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
    	        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
    	          public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
    	            return new Tuple2<Tuple2<Integer, Integer>, Double>(
    	              new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
    	          }
    	        }
    	    ));
    
    System.out.println("#########---Predictions---##########"+predictions.first());
    
    
    JavaRDD<Tuple2<Double, Double>> ratesAndPreds = 
    	      JavaPairRDD.fromJavaRDD(ratings.map(
    	        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
    	          public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
    	            return new Tuple2<Tuple2<Integer, Integer>, Double>(
    	              new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
    	          }
    	        }
    	    )).join(predictions).values();
    
    	    double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
    	      new Function<Tuple2<Double, Double>, Object>() {
    	        public Object call(Tuple2<Double, Double> pair) {
    	          Double err = pair._1() - pair._2();
    	          return err * err;
    	        }
    	      }
    	    ).rdd()).mean();
    	    
    	    System.out.println("Mean Squared Error = " + MSE);

    	    
    	    //predictions.saveAsTextFile("./predictions");
    	    //userProducts.saveAsTextFile("./userProducts");
    	    //ratesAndPreds.saveAsTextFile("./ratesAndPreds");
    	    
    //System.out.println("Users List: " + datauser.first());
    
    System.out.println("Users List:  **********************************************" + ratings.first());
    //System.out.println("Testing Print"+ data.first());
	}
}
