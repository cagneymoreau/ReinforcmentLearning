package Examples;


import DL4JSupport.Display;
import DL4JSupport.FileManager;
import Enviroment.NewGym;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.eclipse.collections.impl.list.Interval;
import org.jfree.data.general.Dataset;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * The VPG and AC I merely ran a single thread of policy updating algos with 2 different
 * reward functions. DL4J has A2C and A3C which are multi environment actor critic tuned for GPU
 * use. So I skipped those
 *
 * Since this is just a practice test case Im not going to complicate it with the parallelization.
 * Although this would be easy to implement by just creating an array of gym.env
 *
 * PPO  Proximal Policy Optimization is really just collecting more data and making smaller updates
 * this is iterative/law of large numbers logic with GPU performance in mind.
 * PPO prevents the loss from swinging wildly out of control basically. If you have
 * run the previous examples you will notice the reward function can collapse and swing wildly. PPO
 * smoothes it out and brings it closer to a monotonic function.
 *
 * Compared to my previous examples I will do this by
 *
 * 1) Using and abitrary trajectory length instead of calling fit on a done=true condition
 *
 * 2) These trajectories datas will be much larger and be sampled from randomly
 *   (Obviously this would be great if I did implement parallel enviroments)
 *
 * 3) Surrogate Policy Loss - Instead of calculating the loss as in regular policy gradient. I will use the PPO loss function
 *  LCLIP(Q)=E^t[min(rt(Q)A^t,clip(rt(Q),1−∈,1+∈)A^t)]
 *
 *  Lclip(Q) = Et[min(???)] This asks which number is smaller of the two. We want less loss
 *  so we have a smoother gradient
 *
 *  The rt(Q) seen twice above is the new network output divided by the old network output.
 *  The its multiplied by your advantage function or clipped.
 *
 *  4) Another upgrade is the generalized advantage estimation GAE
 *
 */

public class PPO {


    private static String save = "D:\\Dropbox\\Apps\\RL\\";
    private static String actorName = "actor_CartPole.dl4j";
    private static String criticName = "critic_CartPole.dl4j";
    public static String game = "CartPole-v1";
    private static String benchmark = "ppo_bench.csv";
    private static double gamma = .99; //discount rate
    private static double lambda = .95; // GAE
    private static double critic_loss = .5; //scal critic loss
    int totalActions = 0;
    int stateSpace = 0;
    int ppo_train_epochs = 8;

    Random rand = new Random(1234);


    public static void main(String[] args)
    {

        PPO ppo = new PPO();

        ComputationGraph actor = null;
        ComputationGraph critic = null;

        try {

            actor = ComputationGraph.load(new File(save + actorName), true);
            critic = ComputationGraph.load(new File(save + criticName), true);

        }catch (Exception e){
            System.err.println("No existing file!");
        }

        actor =  ppo.train(1000000, 320,32, actor, critic);

        Display.plotSimpleSingle(save+benchmark, "PPO CARTPOLE");


        //ppo.test(actor);

    }



    public ComputationGraph train(int epoch, int trajectory, int batchSize, ComputationGraph actor, ComputationGraph critic)
    {

        if (trajectory % batchSize != 0) {
            System.err.println("Trajectory should be divisible by batch size");
            return null;
        }


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


        int cumReward = 0;
        boolean done = true;
        INDArray dist;
        INDArray oldObs = null;

        for (int i = 0; i < epoch; i++) {

            /**
             */
            ArrayList<INDArray> stateLs = new ArrayList<>();
            ArrayList<INDArray> probsLs = new ArrayList<>();
            ArrayList<Integer> actionLs = new ArrayList<>();
            ArrayList<INDArray> rewardLs = new ArrayList<>();
            ArrayList<Boolean> doneLs = new ArrayList<>();

            //long start = System.currentTimeMillis();

            int trajCount = 0;
            while(trajCount < trajectory){
                if (done){

                    framesList.add((double) framecount);
                    rewardEarned.add((double) cumReward);
                    System.out.println("Reward: " + cumReward);
                    if (cumReward == 500) break;

                    stepResponse = newGym.reset();
                    oldObs = stepResponse.getObservation();
                    cumReward = 0;

                }

                dist = actor.outputSingle(oldObs);

                int action = new EnumeratedIntegerDistribution(actionSpace, dist.dup().data().asDouble()).sample();

                stepResponse = newGym.step(action);

                stateLs.add(oldObs);
                probsLs.add(dist);
                actionLs.add(action);
                rewardLs.add(stepResponse.getReward());
                cumReward += stepResponse.getReward().getInt(0);

                done = stepResponse.isDone();
                doneLs.add(done);
                oldObs =stepResponse.getObservation();

                trajCount++;
                framecount++;
            }

            //long finish = System.currentTimeMillis();
            //System.out.println((finish - start)/ 1000);
            if (cumReward == 500) break;


            stateLs.add(oldObs); // we need this future stat for V(s) value

            System.out.println(" -- fit --");
            for (int j = 0; j < ppo_train_epochs; j++) {

                fit(actor, critic, batchSize, stateLs, actionLs, probsLs, doneLs, rewardLs);

            }


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


    public void fit(ComputationGraph actor, ComputationGraph critic, int batchSize, ArrayList<INDArray> state,
                    ArrayList<Integer> action, ArrayList<INDArray> probs, ArrayList<Boolean> done,
                    ArrayList<INDArray> reward)
    {
        INDArray states = state.get(0);

        ArrayList<INDArray> list = calcGAE(critic, state, action, reward, done);
        INDArray advantage = list.get(0);
        INDArray returns = list.get(1);
        //INDArray advantage = calcTD(critic, state, action, reward, done);
        //state.remove(state.size()-1); //this was for td calcs only

        INDArray doneArr = Nd4j.zeros(done.size(), 1);
        //prob array
        INDArray probArr = probs.get(0);
        //one hot actions
        INDArray actions = Nd4j.zeros(action.size(), totalActions);
        for (int i = 0; i < action.size(); i++) {
            actions.putScalar(i, action.get(i), 1);
            //probArr.put(i, probs.get(i));

            if (done.get(i)) doneArr.putScalar(i, 1);

            if (i != 0)
            {
                states = Nd4j.concat(0, states, state.get(i));
                probArr = Nd4j.concat(0, probArr, probs.get(i));
            }

        }

        DataSet actords = new DataSet();

        INDArray temp = Nd4j.hstack( actions, advantage);
        INDArray temp2 = Nd4j.hstack(probArr, doneArr);

        temp = Nd4j.hstack(temp, temp2); //actions, advantage, pobs, done



        actords.setLabels(temp);
        actords.setFeatures(states.dup());

        actords.shuffle(rand.nextInt());
        List<DataSet> l = actords.batchBy(1);




        DataSet criticDs = new DataSet();

        criticDs.setFeatures(states);
        criticDs.setLabels(returns);

        criticDs.shuffle();
        List<DataSet> l2 = criticDs.batchBy(1);

        DataSetIterator actIter = new ListDataSetIterator<>(l, 32);
        DataSetIterator critIter = new ListDataSetIterator<>(l2, 32);
        critic.fit(critIter);
        actor.fit(actIter);


        /**
         * // TODO: 9/14/2020  compare
         * I could have just used fit one list after another. And I did try this and was
         * successful training my critic network for the total batch size followed by the
         * actor. However most examples show a 1 to 1 train gradient calc.
         * So my minibach example below is slightly closer to that
         */
        /*
        for (int i = 0; i < l2.size(); i+= batchSize) {

            List<DataSet> actData = new ArrayList<>(l.subList(i, i + batchSize));
            List<DataSet> critData = new ArrayList<>(l2.subList(i, i + batchSize));

            DataSetIterator actIter = new ListDataSetIterator<>(actData);
            DataSetIterator critIter = new ListDataSetIterator<>(critData);

            critic.fit(actIter);
            actor.fit(critIter);


        }


         */



    }

    public ArrayList<INDArray> calcGAE(ComputationGraph critic, ArrayList<INDArray> state,
                                  ArrayList<Integer> action, ArrayList<INDArray> reward, ArrayList<Boolean> done)
    {
        INDArray returns = Nd4j.zeros(action.size(), 1);
        INDArray advantage;
        INDArray values = Nd4j.zeros(action.size(), 1);

        double gae = 0.0;

        for (int i = action.size()-1; i >= 0; i--) {

            INDArray predQ = critic.outputSingle(state.get(i));
            values.put(i, predQ);
            INDArray futQ = critic.outputSingle(state.get(i+1));

            int mask = done.get(i) ? 0 : 1; //if done use reward only


            double delta = reward.get(i).getDouble(0) + gamma * futQ.getDouble(0)
                    * mask - predQ.getDouble(0);
            gae = delta + gamma * lambda * mask * gae;

                returns.put(i, predQ.addi(gae));

        }


        /*
        for (int i = 0; i < action.size(); i++) {

            INDArray predQ = critic.outputSingle(state.get(i));
            INDArray futQ = critic.outputSingle(state.get(i+1));

            int mask = done.get(i) ? 1 : 0; //if done use reward only

            double  adv = reward.get(i).getDouble(0) +
                    (gamma * futQ.getDouble(0) - predQ.getDouble(0)) * (1 - mask);

            advantage.putScalar(i, adv);

        }

         */
        ArrayList<INDArray> vals = new ArrayList<>();
        advantage = returns.sub(values);
        vals.add(advantage);
        vals.add(returns);

        return vals;
    }

    // TODO: 9/11/2020
    public INDArray calcTD(ComputationGraph critic, ArrayList<INDArray> state,
                           ArrayList<Integer> action, ArrayList<INDArray> reward, ArrayList<Boolean> done)
    {

        INDArray val = Nd4j.zeros(action.size(), 1);

        for (int i = 0; i < action.size(); i++) {

            INDArray futQ = critic.outputSingle(state.get(i+1));

            int mask = done.get(i) ? 1 : 0; //if done use reward only

            double  adv = reward.get(i).getDouble(0) +
                    gamma * (1- mask) * futQ.getDouble(0);

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

        confBuild.addLayer("action", new OutputLayer.Builder(new PPO.CustomLoss()).nOut(actions).activation(Activation.SOFTMAX).build(), "policy_2");


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





    class CustomPPOLoss implements ILossFunction{

        @Override
        public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
            return 0;
        }

        @Override
        public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
            return null;
        }

        // labels = actions, advantage, pobs, done
        @Override
        public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

            double lossClip = .2;
            double entropyLoss = 5e-3;

            INDArray actions = labels.getColumns(0,1);
            INDArray advantages = labels.getColumns(2);
            INDArray probs = labels.getColumns(3,4);
            INDArray done = labels.getColumns(5);

            INDArray output = activationFn.getActivation(preOutput.dup(), true).add(1e-5); //putput without zero

            INDArray oldLogProbs = Transforms.log(probs, true);
            INDArray newLogProbs = Transforms.log(output, true);


            INDArray ratio = newLogProbs.div(oldLogProbs);

            //
            //INDArray ratio = (newLogProbs.sub(oldLogProbs));
            //ratio = Transforms.pow(ratio, 0);


            INDArray surr1 = ratio.mul(advantages);
            //INDArray min1 = Transforms.min(ratio.dup(), 1 + lossClip);
            INDArray minMax = Transforms.max(Transforms.min(ratio.dup(), 1 + lossClip), 1 - lossClip);
            INDArray surr2 = minMax.mul(advantages);

            //INDArray loss = surr1.minNumber().doubleValue() <  surr2.minNumber().doubleValue() ? surr1 : surr2;
            INDArray loss = Transforms.min(surr1, surr2);

            loss.muli(actions).negi();

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

            INDArray actions = labels.getColumns(0,1);
            INDArray advantages = labels.getColumns(2);
            INDArray probs = labels.getColumns(3,4);
            INDArray done = labels.getColumns(5);

            INDArray labelsR = actions.mul(advantages);


            INDArray output = activationFn.getActivation(preOutput.dup(), true).add(1e-5); //putput without zero
            INDArray logOut = Transforms.log(output, true); //lnpi(a|s)
            INDArray loss = logOut.mul(labelsR); // A(t) * lnpi(a|s)

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
