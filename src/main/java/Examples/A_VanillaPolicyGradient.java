package Examples;

import Enviroment.NewGym;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.eclipse.collections.impl.list.Interval;
import org.eclipse.collections.impl.list.primitive.IntInterval;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;

/**
 *CartPole
 *
 * A policy gradient takes in game state observations and directly outputs a softmaxe for discreete
 * actions or a sigmoid ?? for continuous.
 *
 * The difference between policy gradient and q learning is Q learning approximates the future rewards. Policy gradient
 *
 *
 */

public class A_VanillaPolicyGradient {

    private static String save = "D:\\Dropbox\\Apps\\RL\\";
    private static String name = "VPG_CartPole.dl4j";


    public static void main(String[] args)
    {
        A_VanillaPolicyGradient vpg = new A_VanillaPolicyGradient();


        ComputationGraph graph = null;

        try {

            graph = ComputationGraph.load(new File(save + name), true);

        }catch (Exception e){
            System.err.println("No existing file!");
        }


        graph =  vpg.train();

        vpg.test(graph);




    }


    public ComputationGraph train(int epoch,  ComputationGraph graph)
    {

        NewGym newGym =null;
        try {
            newGym = new NewGym("Pong-v0", NewGym.ObservationStyle.continuous, false);
        } catch (RuntimeException e) {
            System.err.println("Is gym_http_api server running?");
        }

        int actions = newGym.actionSpace().getNumber();
        int[] actionSpace = new int[actions];
        for (int i = 0; i < actionSpace.length; i++) {
            actions = i;
        }
        NewGym.StepResponse stepResponse = newGym.reset();
        int input = (int) stepResponse.getObservation().shape()[0];

        if (graph == null){
            graph = getVPGNetwork(input, actions);
        }


        for (int i = 0; i < epoch; i++) {

            int cumReward = 0;
            boolean done = false;
            stepResponse = newGym.reset();
            INDArray dist;
            INDArray oldObs = stepResponse.getObservation();
            ArrayList<INDArray> stateLs = new ArrayList<>();
            ArrayList<Integer> actionLs = new ArrayList<>();
            ArrayList<INDArray> rewardLs = new ArrayList<>();

            while (!done){

            dist = graph.outputSingle(oldObs);

            int action = new EnumeratedIntegerDistribution(actionSpace, dist.dup().data().asDouble()).sample();

            stepResponse = newGym.step(action);

            stateLs.add(oldObs);
            actionLs.add(action);
            rewardLs.add(stepResponse.getReward());
            cumReward += stepResponse.getReward().getInt(0);

            done = stepResponse.isDone();
            oldObs = stepResponse.getObservation();
            }

            fit();

            System.out.println("Reward: ");
            saveNet(graph);
        }

        return graph;
    }

    public void test(ComputationGraph graph)
    {

    }


    public void fit(ComputationGraph graph, ArrayList<INDArray> state, ArrayList<Integer> action, ArrayList<INDArray> reward){







    }


    public ComputationGraph getVPGNetwork(int discObserv, int actions)
    {
        double dropout = 0;

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(.001))
                .weightInit(WeightInit.ONES)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("disc_in");


        confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(32).build(), "disc_in");
        confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(32).build(), "policy_1");

        confBuild.addLayer("action", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nOut(actions).activation(Activation.SOFTMAX).build(), "policy_2");


        confBuild
                .setOutputs("action")
                .setInputTypes(InputType.feedForward(discObserv))
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();

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






}
