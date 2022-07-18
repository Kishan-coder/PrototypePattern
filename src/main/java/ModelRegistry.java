import java.util.HashMap;

//step4: create and populate registry
public class ModelRegistry {
    HashMap<ModelName, MLModel> baseModels = new HashMap<>();

    public void registerModel(MLModel mlModel){
        baseModels.put(mlModel.getName(), mlModel);
    }

    public MLModel retrieveMLModel(ModelName modelName){
        return baseModels.get(modelName);
    }
}
