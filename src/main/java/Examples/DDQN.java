package Examples;

import DL4JSupport.Display;
import DL4JSupport.FileManager;
import DL4JSupport.ImageSaver;
import Enviroment.NewGym;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jcodec.codecs.mjpeg.MCU;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Deep q Learning Explained simply
 * You should understand a simple q learning table and feed forward supervised training first
 *
 * What the DQN network should do for us...
 * Out network takes in a game state with outputs matching the number of actions we can take
 * This network attempts to approximate a function that outputs the maximum possible future rewards for each action. So
 * for example if you have left and right as you only action it would output [.001, 167] 167 is reward for going right .001 probably means death
 * Then we use an argmax function to select the action with the highest future reward
 * Ths lets us play the game by choosing an action that will help us win
 *
 * How do we train?
 * We call network.output() and we will get an array of values representing the
 * estimated maximum future discounted reward of each action as shown above
 * But we don't have labels to calculate loss because we dont know what our best choice actually was or is
 * After all our network is supposed to learn that as its trained.
 * So, what we do is calculate our reward twice. Once on this state and once on the previous state making sure
 * to include any rewards earned/lost in between. So our networks output at a later state is used to do error backprop for earlier states
 * Q(s,a)=r+γ∗maxa'(Q(s'a')) - This is Temporal Difference TD Learning.
 * The formula above Q() funtion represents our network making a prediction. As you can see if our network is trying to predict all future
 * rewards the previous states prediction of the far left should be equal to the resulting reward plus our next prediction.
 * That's where we get our Error from. The γ is a gamma. Because we our future rewards are risky we discount them.
 *
 * https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
 * https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
 * https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html#atari
 *
 * Warning:
 * This cartpole example will take  maybe 50k frames to converge
 * Pong should take a million steps to converge maybe many more
 * other games 40 million +
 *
 */






public class DDQN {


    ImageSaver imageSaver = new ImageSaver();
    private static String game = "CartPole-v1";
    private static String save = "D:\\Dropbox\\Apps\\RL\\";
    private static String name = "ddqncartpole.dl4j";
    private static String benchmark = "ddqn_bench.csv";
    JFrame f = new JFrame();
    int stateSpace = 0;
    Random rand = new Random(123);
    ArrayList<Double> rewardsList = new ArrayList<>();
    ArrayList<Double> framesList = new ArrayList<>();


    public static void main(String[] args){


        DDQN ddqn = new DDQN();

        ComputationGraph graph = null;

        try {

            graph = ComputationGraph.load(new File(save + name), true);

        }catch (Exception e){
            System.err.println("No existing file!");
        }


        graph =  ddqn.train(10, 10000, 8, 100, .99, .9, 1000, graph );

        Display.plotSimpleSingle(save+benchmark, "DDQN CARTPOLE");

        //ddqn.test(graph);


    }

    // TODO: 9/3/2020 Skip frame is every frame neccessary?
    public ComputationGraph train(int epochs, int maxFramesEpsDecay, int batchSize, int updateTarget, double gamma, double epsilon, int maxBuffer, ComputationGraph net_Saved)
    {

        //Build client/server
        NewGym newGym =null;
        try {
            newGym = new NewGym(game, NewGym.ObservationStyle.discreet, false);
        } catch (RuntimeException e) {
            System.err.println("Is gym_http_api server running?");
        }


        //get important data
        double masterEpsilonDecay = (1)/ (double) (epochs);
        int actions = newGym.actionSpace().getNumber();
        NewGym.StepResponse stepResponse = newGym.reset();
        stateSpace = (int) stepResponse.getObservation().shape()[1];


        //Get both networks
        ComputationGraph network = net_Saved;
        ComputationGraph target_network = getDQNNetwork(stateSpace, actions);
        if (net_Saved == null){
            //Get both networks
            network = getDQNNetwork(stateSpace, actions);
        }


        //loop objects
        ArrayList<DDQN.Replay> replays = new ArrayList<>();
        NewGym.StepResponse sp = null;
        INDArray inout = null;
        boolean done = true;
        int trajReward = 0;
        int framecount = 0;
        boolean maxReached = false;

        //epochs
        for (int i = 0; i < epochs; i++) {

            int count = 0;
            double epsilonDecay = epsilon/ maxFramesEpsDecay;
            epsilon -= masterEpsilonDecay;
            double epochEpsi = epsilon;

            while (count < maxFramesEpsDecay) {

                if (done) {
                    sp = newGym.reset();
                    inout = sp.getObservation(); // TODO: 8/27/2020 difference frames or multi frame
                    rewardsList.add((double ) trajReward);
                    framesList.add((double) framecount);
                    System.out.println("Epsilon: " + epochEpsi + "  Rewards: " + trajReward);
                    trajReward = 0;
                    maxReached = checkMaxReached(1, 500);
                    if (maxReached)
                    {
                        System.out.println("max1");
                        break;
                    }
                }

                //get networks reward estimate for each action
                INDArray outRaw = network.outputSingle(inout);

                //select which action has the highest reward
                int action = (int) (long) Nd4j.getExecutioner().execAndReturn(new IMax(outRaw)).getFinalResult();

                //explore vs exploit
                action = exploitExplore(action, actions, epochEpsi);
                epochEpsi -= epsilonDecay;

                //record reward
                NewGym.StepResponse resp = newGym.step(action); // TODO: 8/25/2020 reward scaling
                trajReward += resp.getReward().getInt(0);

                //load replay buffer
                replays.add(new DDQN.Replay(resp, inout, action));
                if (replays.size() == maxBuffer) {
                    replays.remove(rand.nextInt(maxBuffer));  // TODO: 8/25/2020 prioritzed replay
                }

                //prepare next loop
                inout = resp.getObservation();
                done = resp.isDone();


                //perform fitting
                if (count > (batchSize * 4))
                {
                    performFit(replays, batchSize, gamma, network, target_network); //train network
                }

                //update target and save
                if (count % updateTarget == 0){
                    updateTarget(target_network, network); //update other network
                    System.out.println("Target Update");
                    //you will notice more rewards after this message.
                    //its not a bug. longer running games are more likely to trigger
                    //this as its triggered by frame count


                }

                count += 1;
                framecount++;

            }
            //saveNet(network); //save our network
            System.out.println("Epoch: " + i);

            if (maxReached)
            {
                System.out.println("max2");
                break;
            }
        }

        ArrayList<ArrayList<Double>> rewardsatFrame = new ArrayList<>();
        rewardsatFrame.add(framesList);
        rewardsatFrame.add(rewardsList);
        FileManager.saveBenchMark(rewardsatFrame, save+benchmark);



        return network;

    }



    public boolean checkMaxReached(int req, int max)
    {
        if (rewardsList.size() < req) return false;

        int index = rewardsList.size() - req;

        boolean complete = true;

        for (int i = 0; i < req; i++) {
            if (rewardsList.get(index) < max){
                complete = false;
            }
            index++;
        }

        return complete;
    }


    public void performFit(List<DDQN.Replay> replays, int batchSize, double gamma, ComputationGraph net, ComputationGraph target)
    {
        //pieces of our minibatch
        DataSet batch = new DataSet();
        INDArray features = Nd4j.zeros(1);
        INDArray labels = Nd4j.zeros(1);

        for (int i = 0; i < batchSize; i++) {

            int index = rand.nextInt(replays.size());

            INDArray predQ = net.outputSingle(replays.get(index).prevObs()); //get our prediction
            INDArray futNetQ = net.outputSingle(replays.get(index).nextObs());

            //use the final reward if game over or the max reward from the target net
            double label;
            int action;
            int multi =  replays.get(index).done() ? 0 : 1;

            if (replays.get(index).done())
            {
                int h = 19;
            }


            action = (int) (long) Nd4j.getExecutioner().execAndReturn(new IMax(predQ)).getFinalResult();

            INDArray futureQ = target.outputSingle(replays.get(index).nextObs());

            //predictedQ = reward + (gamma * futureQ)
            label = replays.get(index).reward() + gamma * ( (double) futureQ.maxNumber() * multi); //get value
            // TODO: 8/24/2020 smoothing & error clipping

            predQ.putScalar(0, action, label); //place the correct reward at correct index leave all else






            if (i == 0){
                features = replays.get(index).prevObs();
                labels = predQ;
            }else{

                features = Nd4j.concat(0, features, replays.get(index).prevObs());
                labels = Nd4j.concat(0, labels, predQ );
            }


        }

        batch.setFeatures(features);
        batch.setLabels(labels);

        net.fit(batch);

    }


    public void test(ComputationGraph graph)
    {

        NewGym newGym =null;
        try {
            newGym = new NewGym(game, NewGym.ObservationStyle.discreet, true);
        } catch (RuntimeException e) {
            System.err.println("Is gym_http_api server running?");
        }

        NewGym.StepResponse sp = newGym.reset();
        INDArray inout = sp.getObservation();
        boolean done = false;

        while(!done){

            INDArray outRaw = graph.outputSingle(inout); //input prev observation

            int action = (int) (long)  Nd4j.getExecutioner().execAndReturn(new IMax(outRaw)).getFinalResult(); //get best action

            NewGym.StepResponse resp = newGym.step(action);

            done = resp.isDone();

            try{
                TimeUnit.MILLISECONDS.sleep(33);
            }catch (Exception e){
                e.printStackTrace();
            }


        }

    }



    public int exploitExplore(int action, int actions, double epsilon)
    {
        if (rand.nextDouble() < epsilon){
            return rand.nextInt(actions);
        }
        return action;
    }



    public void updateTarget(ComputationGraph target, ComputationGraph net)
    {

        //extract the network params to gen so that it has its new learned strategy
        for (int i = 0; i < net.getLayers().length; i++) {
            target.getLayer(i).setParams(net.getLayer(i).params());
        }


    }


    public ComputationGraph getDQNNetwork(int observspace, int actions)
    {
        double dropout = 0;

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(.001))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("input");


        confBuild.addLayer("decision", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(512).build(), "input");

        confBuild.addLayer("action", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nOut(actions).activation(Activation.IDENTITY).build(), "decision");


        confBuild
                .setOutputs("action")
                .setInputTypes(InputType.feedForward(observspace))
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();


        return net;



    }



    public void saveNet(ComputationGraph net){

        try {
            net.save(new File(save, name ));
        }catch (Exception e){
            e.printStackTrace();
        }

    }



    public class Replay
    {

        NewGym.StepResponse response;
        int action;
        INDArray old_obs;

        // response recieved, obs we saw and action we took
        public Replay(NewGym.StepResponse response, INDArray obs, int action)
        {
            this.response = response;
            this.action = action;
            this.old_obs = obs;

        }

        public double reward()
        {
            return response.getReward().getDouble(0);
        }

        public INDArray nextObs()
        {
            return response.getObservation();
        }

        public boolean done()
        {
            return response.isDone();
        }

        public INDArray prevObs()
        {
            return old_obs;
        }

        public int action()
        {
            return action;
        }

    }







}
