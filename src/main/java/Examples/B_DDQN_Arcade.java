package Examples;

import DL4JSupport.Display;
import DL4JSupport.ImageSaver;
import Enviroment.NewGym;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;


/**
 * Deep q Learning Explained simply
 * You should understand a simple q learning table and feed forward supervised training first
 *
 * What the DQN network should do for us...
 * Out network takes in a game state and with outputs matching the number of actions we can take
 * This network attempts to approximate a function that outputs the maximum possible future rewards for each action. So
 * for example if you have left and right as you only action it would output [.001, 167] 167 is reward for going right .001 probably means death
 * Then we use an argmax function to select the action with the hightest future reward
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
 * Warning: Pong should take a million steps to converge, other games 40 million
 *
 */


     //// TODO: 8/25/2020 rewards graph  & loss charts etc
        // TODO: 8/25/2020 history processing



public class B_DDQN_Arcade {

        ImageSaver imageSaver = new ImageSaver();
    private static String save = "D:\\Dropbox\\Apps\\RL\\";
    private static String name = "FL.dl4j";
    JFrame f = new JFrame();


    Random rand = new Random(123);

    public static void main(String[] args)
    {
       B_DDQN_Arcade BDDQNArcade_ = new B_DDQN_Arcade();


        ComputationGraph graph = null;

        try {

            graph = ComputationGraph.load(new File(save + name), true);

        }catch (Exception e){
            System.err.println("No existing file!");
        }


             graph =  BDDQNArcade_.train(1000000, .9, .9, 10,
                500, 1000, 3, graph);




        BDDQNArcade_.test(graph);


    }



    public ComputationGraph train(
        int train_Frames,
        double epsilon, //start high and slowly deteriorate
        double gamma,
        int miniBatch,
        int target_update_Delay,
        int maxBuffer,
        int skipframe,  /// TODO: 8/24/2020
        ComputationGraph net_Old)
    {
        /*
        f.setSize(200,200);
        ArrayList<Integer> test = new ArrayList<>();
        test.add(0);

        for (int i = 0; i < 10; i++) {
            showChart(test);
            test.add(rand.nextInt(200));
        }

         */

        NewGym newGym =null;
        try {
            newGym = new NewGym("Pong-v0", NewGym.ObservationStyle.continuous, false);
        } catch (RuntimeException e) {
            System.err.println("Is gym_http_api server running?");
        }

        double epsilonDecay = (epsilon- .1)/train_Frames;
        int actions = newGym.actionSpace().getNumber();

        //Get both networks
        ComputationGraph network = net_Old;
        ComputationGraph target_network = getDQNNetwork(actions);

        if (net_Old == null){

            //Get both networks
            network = getDQNNetwork(actions);


        }



        //collect experience replays and train
        ArrayList<Replay> replays = new ArrayList<>();

        NewGym.StepResponse sp = null;
        INDArray inout = null;
        boolean done = true;
        ArrayList<Integer> rewards = new ArrayList<>();
        Integer accumRewards = 0;

        for (int i = 0; i < train_Frames; i++) {
            //Map<String, INDArray> outB = network.feedForward(inout,network.getLayers().length-1, false);
            if (done){
                 sp = newGym.reset();
                 inout = scaleImage(sp.getObservation());
                 rewards.add(accumRewards);
                System.out.println("rewards:  " + accumRewards);
                accumRewards  = 0;
                 //showChart(rewards);
            }
            // TODO: 8/27/2020 differenc frames or multi frame
            INDArray outRaw = network.outputSingle(inout); //input prev observation

            int action = (int) (long)  Nd4j.getExecutioner().execAndReturn(new IMax(outRaw)).getFinalResult(); //get best action

            //explore vs exploit
            action = exploitExplore(action, actions, epsilon);
            epsilon = epsilon - epsilonDecay;

            //load replay buffer
            NewGym.StepResponse resp = newGym.step(action); // TODO: 8/25/2020 reward scaling
            accumRewards += resp.getReward().getInt(0);
            /*
            if (resp.getReward().getInt(0) > 0 || resp.isDone())
            {
                System.out.println();// TODO: 8/25/2020 set debugger here
            }

             */
            replays.add(new Replay(resp, inout, action));
            if (replays.size() == maxBuffer){
                // TODO: 8/25/2020 prioritzed replay
                replays.remove(rand.nextInt(maxBuffer));
            }

            if (i > miniBatch * 4){
                for (int j = 0; j < 3; j++) {
                    trainstep(replays, miniBatch, gamma, network, target_network);
                }

            }

            if (i % target_update_Delay == 0)
            {
                updateTarget(target_network, network);
                saveNet(network);
            }

            inout = scaleImage(resp.getObservation()); //scale for next iteration

            done = resp.isDone();




            /*
            if (i % 100 == 0) {
                imageSaver.saveImage(resp.getObservation(), "o_" + String.valueOf(i + 1));
                imageSaver.saveImage(inout, "g_" + String.valueOf(i + 1));
            }
             */
        }

        return network;

    }

    public void test(ComputationGraph network)
    {
        NewGym newGym =null;
        try {
            newGym = new NewGym("SpaceInvaders-v0", NewGym.ObservationStyle.continuous, true);
        } catch (RuntimeException e) {
            System.err.println("Is gym_http_api server running?");
        }

        NewGym.StepResponse sp = newGym.reset();
        INDArray inout = scaleImage(sp.getObservation());
        boolean done = false;

        while(!done){

            INDArray outRaw = network.outputSingle(inout); //input prev observation

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

    public void trainstep(List<Replay> replays, int batchSize, double gamma, ComputationGraph net, ComputationGraph target)
    {
        //pieces of our minibatch
        DataSet batch = new DataSet();
        INDArray features = Nd4j.zeros(1);
        INDArray labels = Nd4j.zeros(1);

        for (int i = 0; i < batchSize; i++) {

            int index = rand.nextInt(replays.size());

            INDArray pred = net.outputSingle(replays.get(index).prevObs()); //get our prediction

            //use the final reward if game over or the max reward from the target net
            double label = replays.get(index).reward();
            int action = replays.get(index).action();

            if (!replays.get(index).done()){

                INDArray out = target.outputSingle(replays.get(index).nextObs());
                // TODO: 8/24/2020 smoothing & error clipping
                label = replays.get(index).reward() + gamma * ( (double) out.maxNumber()); //get value

                action = (int) (long) Nd4j.getExecutioner().execAndReturn(new IMax(out)).getFinalResult(); //get best action

                pred.putScalar(0, action, label); //place the correct reward at correct index leave all else

            }




        if (i == 0){
            features = replays.get(index).prevObs();
            labels = pred;
        }else{

            features = Nd4j.concat(0, features, replays.get(index).prevObs());
            labels = Nd4j.concat(0, labels, pred );
        }


        }

        batch.setFeatures(features);
        batch.setLabels(labels);

        net.fit(batch);

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


    //3x210x160
    public INDArray scaleImage(INDArray image)
    {
        // preporc
        //image = image.div(255);
        image = image.div(3);


        //gray
        INDArray gray = image.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
            .add(image.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()))
            .add(image.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all()));

        gray = gray.get(NDArrayIndex.interval(35, 195), NDArrayIndex.all()); //crop top of screen and bit of bottom

        //downsample
        gray = gray.get(NDArrayIndex.interval(0,2,160),NDArrayIndex.interval(0,2,160));

        gray = gray.reshape(1,1,80,80);
        //imageSaver.saveImage(gray, "gray_" + imageSaver.getCount());
        return gray; //1x80x80

    }


    public void showChart(ArrayList<Integer> data)
    {
        JPanel panel = Display.plotSimpleSingle(data, "reward");

        //f.setSize(2000,200);
        //f.removeAll();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        //f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);

    }



    public ComputationGraph getDQNNetwork(int actions)
    {
        double dropout = 0;

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(.001))
            .weightInit(WeightInit.ONES)
            .l2(1e-4)
            .graphBuilder()
            .addInputs("screen_in");

        confBuild.addLayer("cnn_1", new ConvolutionLayer.Builder().kernelSize(8,8).padding(0,0).stride(4,4).nIn(1).dropOut(dropout).nOut(32).activation(Activation.LEAKYRELU).build(), "screen_in");
        confBuild.addLayer("cnn_2", new ConvolutionLayer.Builder().kernelSize(4,4).padding(0,0).stride(2,2).nIn(32).dropOut(dropout).nOut(32).activation(Activation.LEAKYRELU).build(), "cnn_1");
        confBuild.addLayer("cnn_3", new ConvolutionLayer.Builder().kernelSize(3,3).padding(0,0).stride(1,1).nIn(32).dropOut(dropout).nOut(32).activation(Activation.LEAKYRELU).build(), "cnn_2");

        confBuild.addLayer("decision", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(512).build(), "cnn_3");

        confBuild.addLayer("action", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nOut(actions).activation(Activation.IDENTITY).build(), "decision");


        confBuild
            .setOutputs("action")
            .setInputTypes(InputType.convolutional(80,80,1))
            .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();

        /*
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        net.setListeners(new StatsListener(statsStorage));

         */

        net.setListeners(new ScoreIterationListener(100));
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
            return scaleImage(response.getObservation());
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
