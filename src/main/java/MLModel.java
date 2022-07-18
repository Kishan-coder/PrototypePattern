import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
//step1: create class
public class MLModel implements ObjectClonable{
    private ModelName name;
    private String description;
    private float trainingSplit;
    private float validationSplit;
    private int epochs;
    private double alpha;
    protected String beta;

    //step2: implement clone
    @Override
    public MLModel clone() {
        return new MLModel(name, description, trainingSplit, validationSplit, epochs, alpha, beta);
    }
}
