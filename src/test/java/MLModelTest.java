import com.sun.javafx.sg.prism.NGShape;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class MLModelTest {

    ModelRegistry modelRegistry = new ModelRegistry();

    @Before
    public void setUp(){
        modelRegistry.registerModel(new MLModel(ModelName.LeafTypeDetect, "Here we detect type of plant from leaves using Deep Learning",
                0.2f, 0.7f, 5, 1, "aTest"));
        modelRegistry.registerModel(new MLModel(ModelName.EmotionDetect, "Here we detect human emotions using openCV Libs",
                0.3f, 0.7f, 25, 13, "aTest2"));
        modelRegistry.registerModel(new MLModel(ModelName.MovieRecommend, "Here we recommend movies based on collaborative filtering",
                0.22f, 0.7f, 4, 21, "aTest3"));
    }

    @Test
    public void testMLModelCreation(){
        MLModel model = new MLModel(ModelName.LeafTypeDetect, "Here we detect type of plant from leaves using Deep Learning",
                0.2f, 0.7f, 5, 1, "aTest");
        Assert.assertTrue("alpha should be 1 if object creation successful", model.getAlpha() == 1);
    }

    @Test
    public void testMLModelCreation2(){
        MLModel model = new MLModel(ModelName.LeafTypeDetect, "Here we detect type of plant from leaves using Deep Learning",
                0.2f, 0.7f, 5, 1, "aTest");
        MLModel cloneofAbove = model.clone();
        Assert.assertTrue("alpha not 1 => cloning Unsuccessful", cloneofAbove.getAlpha() == 1);
        Assert.assertFalse("Clone is same than actual type-1", cloneofAbove.equals(model));
    }

    //step3: cone models using prototype
    @Test
    public void testMultipleCopies(){
        MLModel prototypeModel = new MLModel(ModelName.LeafTypeDetect, "Here we detect type of plant from leaves using Deep Learning",
                0.2f, 0.7f, 5, 1, "aTest");
        List<MLModel> listOfCopies = new ArrayList<>();
        for (int i=0;i<10;i++){
            MLModel clonedModel = prototypeModel.clone();
            clonedModel.setAlpha(2*prototypeModel.getAlpha());
            clonedModel.setEpochs(1+prototypeModel.getEpochs());
            listOfCopies.add(clonedModel);
        }
        Assert.assertEquals("lenght != 10, an error", 10, listOfCopies.size());
    }

    //step 4*: clone models using prototypes registered in registry
    @Test
    public void testCopiesUsingRegistry(){
        List<MLModel> listOfCopiesUsingPrototypeRegistry = new ArrayList<>();
        //creating 9 clones of type-1
        for (int i = 0 ;i<9 ;i++){
            MLModel prototype1Model = modelRegistry.retrieveMLModel(ModelName.LeafTypeDetect);
            MLModel clonedModel = prototype1Model.clone();
            clonedModel.setAlpha(0.8*clonedModel.getAlpha());
            clonedModel.setBeta(clonedModel.getBeta()+"->"+i);
            listOfCopiesUsingPrototypeRegistry.add(clonedModel);
        }
        //creating 9 clones of type-2
        for (int i = 0 ;i < 9 ;i++){
            MLModel prototype1Model = modelRegistry.retrieveMLModel(ModelName.LeafTypeDetect);
            MLModel clonedModel = prototype1Model.clone();
            clonedModel.setAlpha(4.8*clonedModel.getAlpha());
            clonedModel.setBeta(clonedModel.getBeta()+"->"+i*5);
            listOfCopiesUsingPrototypeRegistry.add(clonedModel);
        }

        Assert.assertEquals("size != 18 means issue with above list", 18, listOfCopiesUsingPrototypeRegistry.size());
    }
}
