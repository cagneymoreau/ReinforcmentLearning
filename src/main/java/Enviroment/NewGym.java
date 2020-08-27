package Enviroment;

import org.deeplearning4j.gym.ClientUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NewGym {



    private final static String URL = "http://127.0.0.1:5000/";
    String envId;
    ObservationStyle observationStyle;
    String instance;

    public enum ObservationStyle { discreet, continuous}

    ActionSpace actionSpace;


    boolean render;

    public NewGym(String game, ObservationStyle observationStyle, boolean render)
    {
        envId = game;
        this.observationStyle = observationStyle;
        this.render = render;
        post_envs();

    }

    //env
    private void post_envs()
    {
        JSONObject body = new JSONObject().put("env_id", envId);
        JSONObject reply = ClientUtils.post(URL + "v1/envs/", body).getObject();

        String instanceID = "";
        try{
            instanceID = reply.getString("instance_id");
        }catch (JSONException e){
            e.printStackTrace();
        }

        instance = instanceID;

    }

    //reset
        public StepResponse reset()
        {


            JSONObject body = new JSONObject().put("env_id", envId);
            JSONObject reply = ClientUtils.post(URL + "v1/envs/" + instance + "/reset/", body).getObject();

            StepResponse stepResponse = new StepResponse();

            JSONArray  observ = reply.getJSONArray("observation");

            setStepResponse(observ, stepResponse);

            return stepResponse;
        }

    //actionspace
    public ActionSpace actionSpace()
    {
        if (actionSpace == null) {

            //JSONObject body = new JSONObject().put("env_id", envId);
            JSONObject reply = ClientUtils.get(URL + "v1/envs/" + instance + "/action_space/");

            JSONObject info = reply.getJSONObject("info");
            String type = info.getString("name");

            if (type.equals("Discrete")) {

                int y = info.getInt("n");
                actionSpace = new ActionSpace(type, y);

            } else {
                throw new RuntimeException("cannot handle continous actionspace");
            }
        }

        return actionSpace;

    }


    //step
    public StepResponse step(int action)
    {

        JSONObject body = new JSONObject().put("env_id", envId)
            .put("action", action)
            .put("render", render);
        JSONObject reply = ClientUtils.post(URL + "v1/envs/" + instance + "/step/", body).getObject();

        StepResponse stepResponse = new StepResponse();

        JSONArray  observ = reply.getJSONArray("observation");
        setStepResponse(observ, stepResponse);

        stepResponse.getReward().putScalar(0, reply.getDouble("reward"));
        stepResponse.setDone(reply.getBoolean("done"));
        stepResponse.setInfo(reply.getJSONObject("info"));

        return stepResponse;

    }


    //sample
    public int sample()
    {
        //JSONObject body = new JSONObject().put("env_id", envId);
        JSONObject reply = ClientUtils.get(URL + "v1/envs/" + instance + "/action_space/sample");

        int a = reply.getInt("action");
        return  a;

    }

    //stepResponse space



    //start

    //close

    //upload

    //shutdown


    private void setStepResponse(JSONArray arr, StepResponse stepResponse)
    {
        if (observationStyle == ObservationStyle.discreet){

            INDArray newObs = Nd4j.zeros(arr.length());


            for (int i = 0; i < arr.length(); i++) {
                newObs.putScalar(i, arr.getDouble(i));
            }

            stepResponse.setObservation(newObs);

        }else
            //we have some rank 3 array
            {
                int height = arr.length();
                int width = arr.getJSONArray(0).length();

                INDArray newObs = Nd4j.zeros(3,height,width);


                for (int i = 0; i < height; i++) {

                    JSONArray row = arr.getJSONArray(i);

                    for (int j = 0; j < row.length(); j++) {

                        JSONArray rgb = row.getJSONArray(j);

                        newObs.putScalar(0,i,j, rgb.getDouble(0));
                        newObs.putScalar(1,i,j, rgb.getDouble(1));
                        newObs.putScalar(2,i,j, rgb.getDouble(2));
                    }

                }

                stepResponse.setObservation(newObs);
                //imageSaver.saveImage(newObs, "set_" + imageSaver.getCount());

        }


    }


    public class ActionSpace
    {

        int n;
        String type;


        public ActionSpace(String type, int n)
        {
            this.n = n;
            this.type = type;

        }

        public int getNumber()
        {
            return n;
        }

        public String getType()
        {
            return type;
        }

    }


    public class StepResponse
    {
        //These are not meant to maintai or store state but merely memory efficiency
        INDArray observation;
        INDArray reward = Nd4j.zeros(1);
        boolean done;
        JSONObject info;



        public StepResponse()
        {

        }

        public INDArray getObservation() {
            return observation;
        }

        public void setObservation(INDArray observation) {
            this.observation = observation;
        }


        public INDArray getReward() {
            return reward;
        }

        public void setReward(INDArray reward) {
            this.reward = reward;
        }

        public boolean isDone() {
            return done;
        }

        public void setDone(boolean done) {
            this.done = done;
        }

        public JSONObject getInfo() {
            return info;
        }

        public void setInfo(JSONObject info) {
            this.info = info;
        }
    }


}
