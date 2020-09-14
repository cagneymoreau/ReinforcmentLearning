package Examples;


import DL4JSupport.Display;
import DL4JSupport.FileManager;
import Enviroment.NewGym;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.ArrayList;

/**
 * Actor critic is a combination of the policy gradient and the Q learning.
 * The policy gradient portion is basically identical in that we iteratively
 * perform the best actions and then reward those actions based on how good or bad they are.
 *
 * But instead of using the rewards from an entire episode with that custom rewards function
 * we use a NN as a rewards estimator. This means our actor is our normal policy gradient
 * but now we add a critic via the extra network to return our rewards
 *
 * Unlike the DQN our critic NN does not return a vector for each action representing Q(s|a) or the
 * future rewards of each action. Instead it returns a scalar V(s) which is the value of a current state.
 *
 * Then we can calc our rewards in a familiar TD calc. The V(s) is our critic NN output.
 * Training Label for our actor policy NN = Reward + * (gamma * V(s+1)) - V(s)
 *
 * Training label for our critic NN is = reward + (gamma * V(s+1))
 *
 *  The advatage of this method is not requiring full game play to get a reward and
 *  have a vector od a single scalar as output from our value function
 *
 */

public class ActorCritic {


    private static String save = "D:\\Dropbox\\Apps\\RL\\";
    private static String actorName = "actor_CartPole.dl4j";
    private static String criticName = "critic_CartPole.dl4j";
    public static String game = "CartPole-v1";
    private static String benchmark = "ac_bench.csv";

    private static double gamma = .99; //discount rate
    int totalActions = 0;
    int stateSpace = 0;



    public static void main(String[] args)
    {

       ActorCritic actorCritic = new ActorCritic();

        ComputationGraph actor = null;
        ComputationGraph critic = null;

        try {

            actor = ComputationGraph.load(new File(save + actorName), true);
            critic = ComputationGraph.load(new File(save + criticName), true);

        }catch (Exception e){
            System.err.println("No existing file!");
        }

        actor =  actorCritic.train(1000000, actor, critic);

        Display.plotSimpleSingle(save+benchmark, "AC CARTPOLE");

        //actorCritic.test(actor);

    }



    public ComputationGraph train(int epoch, ComputationGraph actor, ComputationGraph critic)
    {
        NewGym newGym = null;
        try{
            newGym = new NewGym(game, NewGym.ObservationStyle.discreet, false);
        } catch (Exception e)
        {
            System.err.println("Is gym_http_api server running?");
        }


        totalActions = newGym.actionSpace().getNumber();
        int[] actionSpace = new int[totalActions];
        for (int i = 0; i < actionSpace.length; i++) {
            actionSpace[i] = i;
        }
        NewGym.StepResponse stepResponse = newGym.reset();
        stateSpace = (int) stepResponse.getObservation().shape()[1];

        if (actor == null){
            actor = getPolicyGradientActor(stateSpace, totalActions);
            critic = getQCritic(stateSpace);
        }

        //benchmark stuff
        int framecount = 0;
        ArrayList<Double>  rewardEarned = new ArrayList<>();
        ArrayList<Double>  framesList = new ArrayList<>();

        for (int i = 0; i < epoch; i++) {


            int cumReward = 0;
            boolean done = false;
            stepResponse = newGym.reset();

            INDArray dist;
            INDArray oldObs = stepResponse.getObservation();

            /** Even though we combine Q into PG we dont use long term replays so
             * I will omit the replays class for a few arrayLists just like VPG.
             */
            ArrayList<INDArray> stateLs = new ArrayList<>();
            ArrayList<Integer> actionLs = new ArrayList<>();
            ArrayList<INDArray> rewardLs = new ArrayList<>();


            while(!done){

                dist = actor.outputSingle(oldObs);

                int action = new EnumeratedIntegerDistribution(actionSpace, dist.dup().data().asDouble()).sample();

                stepResponse = newGym.step(action);

                stateLs.add(oldObs);
                actionLs.add(action);
                rewardLs.add(stepResponse.getReward());
                cumReward += stepResponse.getReward().getInt(0);

                done = stepResponse.isDone();
                oldObs =stepResponse.getObservation();

                framecount++;
            }
            stateLs.add(oldObs);
            fit(actor, critic, stateLs, actionLs, rewardLs);

            framesList.add((double) framecount);
            rewardEarned.add((double) cumReward);
            System.out.println("Reward: " + cumReward);
            //if (cumReward == 500) break;
            if (framecount > 100000) break;
        }

        ArrayList<ArrayList<Double>> lll = new ArrayList<>();
        lll.add(framesList);
        lll.add(rewardEarned);
        FileManager.saveBenchMark(lll, save + benchmark);

        return actor;
    }

    public void test(ComputationGraph actor)
    {

    }


    public void fit(ComputationGraph actor, ComputationGraph critic, ArrayList<INDArray> state,
                    ArrayList<Integer> action, ArrayList<INDArray> reward)
    {
        INDArray states = state.get(0);

        INDArray advantage = calcAdvantage(critic, state, action, reward);

        //one hot actions
        INDArray actions = Nd4j.zeros(action.size(), totalActions);
        for (int i = 0; i < action.size(); i++) {
            actions.putScalar(i, action.get(i), 1);

            if (i != 0)
            {
                states = Nd4j.concat(0, states, state.get(i));
            }

        }

        DataSet actords = new DataSet();

        //INDArray temp = Nd4j.hstack( actions, advantage);
        INDArray temp = actions.mul(advantage);

        actords.setLabels(temp);
        actords.setFeatures(states.dup());

        actor.fit(actords);


        DataSet criticDs = new DataSet();

        criticDs.setFeatures(states);
        criticDs.setLabels(calcTD(critic, state, action, reward));

        critic.fit(criticDs);



    }

    public INDArray calcAdvantage(ComputationGraph critic, ArrayList<INDArray> state,
                              ArrayList<Integer> action, ArrayList<INDArray> reward)
    {
        INDArray advantage = Nd4j.zeros(action.size(),1);

        for (int i = 0; i < action.size(); i++) {

            if (i == action.size()-1)
            {
                advantage.putScalar(i,reward.get(i).getDouble(0));
                continue;
            }

            INDArray predQ = critic.outputSingle(state.get(i));
            INDArray futQ = critic.outputSingle(state.get(i+1));


            double  adv = reward.get(i).getDouble(0) +
                    gamma * futQ.getDouble(0) - predQ.getDouble(0);

            advantage.putScalar(i, adv);

        }

        return advantage;
    }


    public INDArray calcTD(ComputationGraph critic, ArrayList<INDArray> state,
                                  ArrayList<Integer> action, ArrayList<INDArray> reward)
    {
        INDArray val = Nd4j.zeros(action.size(), 1);

        for (int i = 0; i < action.size(); i++) {

            if (i == action.size()-1)
            {
                val.putScalar(i,reward.get(i).getDouble(0));
                continue;
            }

            //INDArray predQ = critic.outputSingle(state.get(i));
            INDArray futQ = critic.outputSingle(state.get(i+1));


            double  adv = reward.get(i).getDouble(0) +
                    gamma * futQ.getDouble(0);

            val.putScalar(i, adv);

        }

        return val;
    }


    //
    public ComputationGraph getPolicyGradientActor(int discObserv, int actions)
    {
        double dropout = 0;

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(.001))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("disc_in");


        confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(32).build(), "disc_in");
        confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(32).build(), "policy_1");

        confBuild.addLayer("action", new OutputLayer.Builder(new ActorCritic.CustomLoss()).nOut(actions).activation(Activation.SOFTMAX).build(), "policy_2");


        confBuild
                .setOutputs("action")
                .setInputTypes(InputType.feedForward(discObserv))
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();

        //net.setListeners(new ScoreIterationListener(100));
        return net;


    }

    //approximate value of a state V(s)
    public ComputationGraph getQCritic(int discObserv)
    {
        double dropout = 0;

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(.001))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("disc_in");


        confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(32).build(), "disc_in");
        confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(32).build(), "policy_1");

        confBuild.addLayer("action", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nOut(1).activation(Activation.IDENTITY).build(), "policy_2");


        confBuild
                .setOutputs("action")
                .setInputTypes(InputType.feedForward(discObserv))
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();

        //net.setListeners(new ScoreIterationListener(100));
        return net;



    }





    class CustomLoss implements ILossFunction{

        @Override
        public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
            return 0;
        }

        @Override
        public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
            return null;
        }

        @Override
        public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
            INDArray output = activationFn.getActivation(preOutput.dup(), true).add(1e-5); //putput without zero
            INDArray logOut = Transforms.log(output, true); //lnpi(a|s)
            INDArray loss = logOut.mul(labels); // A(t) * lnpi(a|s)

            INDArray gradient = activationFn.backprop(preOutput, loss).getFirst();

            return gradient;
        }

        @Override
        public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
            return null;
        }

        @Override
        public String name() {
            return null;
        }
    }



}
