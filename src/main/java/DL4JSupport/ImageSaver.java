package DL4JSupport;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class ImageSaver {

    Java2DNativeImageLoader imageLoader;
    private String save = "D:\\Dropbox\\Apps\\RL\\";
    int count = -1;

    public ImageSaver()
    {
        imageLoader = new Java2DNativeImageLoader();
    }

    public void saveImage(INDArray im, String name)
    {
        BufferedImage bf = imageLoader.asBufferedImage(im);


        File f = new File(save + name +  ".jpg");

        try {
            ImageIO.write(bf, "jpg", f);
        }catch (Exception e)
        {
            System.out.print(" ");
        }


    }

    public int getCount()
    {
        count++;
        return count;
    }


}
