package DL4JSupport;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.dataset.DataSet;

import javax.swing.*;
import java.io.File;
import java.util.List;

public class Display {




    public static JPanel plotSimpleSingle(List<Integer> r, String desc) {
        XYSeriesCollection data = new XYSeriesCollection();


            XYSeries series = new XYSeries(desc);

            //cycle through 3d for number of examples exluding last dimensions overstuffed
            for (int b = 0; b < r.size(); b++) {

                series.add(b, r.get(b));
                //series.add( new Day(day[b], month[b], year[b]) , d.getFeatures().getFloat(b,i,0));

            }
            data.addSeries(series);



        String title = "test";
        String xAxisLabel = "Date";
        String yAxisLabel = "Price";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, data, PlotOrientation.VERTICAL, legend,tooltips,urls);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);

        return  panel;


    }



    public static JPanel plotCSV(String file, String desc) {


        List<Double> r = null;

        try {
             r = FileManager.retreiveColumnFullCSV(file, 0);
        }catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }

        XYSeriesCollection data = new XYSeriesCollection();


        XYSeries series = new XYSeries(desc);

        //cycle through 3d for number of examples exluding last dimensions overstuffed
        for (int b = 0; b < r.size(); b++) {

            series.add(b, r.get(b));
            //series.add( new Day(day[b], month[b], year[b]) , d.getFeatures().getFloat(b,i,0));

        }
        data.addSeries(series);



        String title = "test";
        String xAxisLabel = "Date";
        String yAxisLabel = "Price";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, data, PlotOrientation.VERTICAL, legend,tooltips,urls);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);

        return  panel;


    }


    public static void plotSimpleSingle(String file, String title) {

        List<Double> frame = null;
        List<Double> reward = null;

        try{
                    frame = FileManager.retreiveColumnFullCSV(file, 0);
            reward = FileManager.retreiveColumnFullCSV(file, 1);

                }catch (Exception e)
                {
                    e.printStackTrace();
                }


        XYSeriesCollection data = new XYSeriesCollection();


            XYSeries series = new XYSeries("series");

            //cycle through 3d for number of examples exluding last dimensions overstuffed
            for (int b = 0; b < reward.size(); b++) {

                series.add(frame.get(b), reward.get(b));


            }
            data.addSeries(series);




        String xAxisLabel = "Frame";
        String yAxisLabel = "Reward";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, data, PlotOrientation.VERTICAL, legend,tooltips,urls);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);


    }


}
