package DL4JSupport;

import org.apache.commons.io.IOUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class FileManager {

    private static boolean checkFileExists(String path)
    {
        File f = new File(path);
        if (!f.exists()){
            return false;
        }

        return true;
    }



    public static List<Double> retreiveColumnFullCSV(String path, int column) throws Exception {

        List<String> lines = IOUtils.readLines(new FileInputStream(path), StandardCharsets.UTF_8);
        List<Double> columns = new ArrayList<>();

        for (int i = 1; i < lines.size(); i++) {

            Double d =  Double.valueOf(lines.get(i).split(",")[column]);

            columns.add(d);

        }

        return columns;
    }

    // framecoutn // reward
    public static void saveBenchMark(ArrayList<ArrayList<Double>> data, String path)
    {


        StringBuilder sb = new StringBuilder();

        for (int j = 0; j < data.get(0).size(); j++) {


            sb.append(data.get(0).get(j)).append(",").append(data.get(1).get(j));


            sb.setLength(sb.length()-1);
            sb.append("\n");

        }
        try {
            rewriteFullCSV(path, sb.toString());
        }catch (Exception e){
            e.printStackTrace();
        }


    }


    public  static void rewriteFullCSV(String path, String vals)throws Exception{


        FileWriter filewriter = new FileWriter(path);

        filewriter.append(vals);

        filewriter.flush();
        filewriter.close();

    }



}
