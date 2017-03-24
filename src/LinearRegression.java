import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;

/**
 * Created by Chen on 2017/3/22.
 */
public class LinearRegression {
    public static class Point {
        public double[] values;
        public double reference;

        public Point(double[] values, double reference) {
            this.values = values;
            this.reference = reference;
        }
    }
    public static void main(String[] args) {
        String filename = "E:\\SYSU\\大三\\计应大三下\\计应大三下\\数据挖掘与机器学习\\Project\\data\\save_train.csv";
        String testFileName = "E:\\SYSU\\大三\\计应大三下\\计应大三下\\数据挖掘与机器学习\\Project\\data\\save_test.csv";
        String outFileName = "E:\\SYSU\\大三\\计应大三下\\计应大三下\\数据挖掘与机器学习\\Project\\data\\save_result.csv";
        ArrayList<Point> points;
        ArrayList<Point> testPoints;
        points = ReadDataSet(filename, null);
        double[] theta = InitialTheta();
        double alpha = 0.095;
        double[] nextTheta;
        int iterationTime = 1;
        while (true) {
            double currentCost = Cost(points, theta);
            System.out.println("正在进行第"+iterationTime+"次迭代, Cost："+currentCost+" alpha值为："+alpha);
            nextTheta = GradientDescent(theta, alpha, points);
            double nextCost = Cost(points, nextTheta);
            iterationTime++;
            if (nextCost < currentCost) {
                if(alpha < 0.095)
                    alpha += 0.001;
                theta = nextTheta;
                if (iterationTime % 10000 == 0) {
                    testPoints = ReadDataSet(testFileName, theta);
                    CsvWriter writer = new CsvWriter(outFileName, ',', Charset.forName("GBK"));
                    try {
                        String[] header = {"id", "reference"};
                        writer.writeRecord(header);
                        for (int i = 0; i < testPoints.size(); i++) {
                            String[] contents = {Integer.toString(i), Double.toString(testPoints.get(i).reference)};
                            writer.writeRecord(contents);
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    writer.flush();
                    writer.close();
                }

            } else {
                alpha -= 0.005;
            }
        }

    }

    private static double[] GradientDescent(double[] theta, double alpha, ArrayList<Point> points) {
        double[] tempTheta = new double[theta.length];
        double[] hypothesisDistance = new double[points.size()];
        int size = points.size();
        for (int i = 0; i < size; i++) {
            Point point = points.get(i);
            hypothesisDistance[i] = Hypothesis(point.values, theta) - point.reference;
        }
        for (int j = 0; j < tempTheta.length; j++) {
            tempTheta[j] = theta[j] - alpha*CostDiffOnTheta(points, j, hypothesisDistance);
        }
        return tempTheta;
    }

    private static ArrayList<Point> ReadDataSet(String filename, double[] theta) {
        ArrayList<Point> points = new ArrayList<>();
        try {
            CsvReader reader = new CsvReader(filename, ',', Charset.forName("GBK"));
            //读取数据
            System.out.println("--------------------------------------开始读取数据--------------------------------");
            reader.readHeaders();
            LocalDateTime start = LocalDateTime.now();
            while (reader.readRecord()) {
                double[] values = new double[385];
                double reference;
                for (int i = 0; i < 384; i++) {
                    values[i] = Double.parseDouble(reader.get("value" + i));
                }
                values[384] = 1;//对应的是θ0
                if (theta == null)
                    reference = Double.parseDouble(reader.get("reference"));
                else
                    reference = Hypothesis(values, theta);
                System.out.println("当前读取第" + reader.get("id") + "条数据," + "reference:" + reference);
                points.add(new Point(values, reference));
            }
            LocalDateTime finish = LocalDateTime.now();
            System.out.println("数据读取完毕, 共花费时间：" + Duration.between(start, finish).toMillis() / 1000 + "s");
            System.out.println("--------------------------------------读取数据完成--------------------------------");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return points;
    }

    private static double[] InitialTheta() {
        double[] theta = new double[385];
        for (int i = 0; i < 385; i++) {
            theta[i] = 0.1;
        }
        return theta;
    }

    private static double Hypothesis(double[] x, double[] theta) {
        if (x.length != theta.length) {
            System.out.println("数组长度不一致，请检查");
            return 0.0;
        }
        double h = 0.0;
        for (int i = 0; i < x.length; i++) {
            h += x[i]*theta[i];
        }
        return h;
    }

    private static double Cost(ArrayList<Point> points, double[] theta) {
        double sum = 0.0;
        int size = points.size();
        for (int i = 0; i < size; i++) {
            Point point = points.get(i);
            sum += Math.pow(Hypothesis(point.values, theta)-point.reference, 2);
        }
        return sum/(2*size);
    }

    private static double CostDiffOnTheta(ArrayList<Point> points, int thetaPos, double[] hypothesisDistance) {
        double sum = 0.0;
        int size = points.size();
        for (int i = 0; i < size; i++) {
            Point point = points.get(i);
            sum += hypothesisDistance[i]*point.values[thetaPos];
        }
        return sum/size;
    }
}
