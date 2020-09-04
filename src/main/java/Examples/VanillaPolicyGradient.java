package Examples;

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
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.ArrayList;

/**
 *CartPole
 *
 * I cant subscript the math so look up these formulas elsewhere or they wont make sense
 * try sutton & barto for full text or llilianweng.github.io for more concise
 *
 *  Idiot/ Intuitive Description:
 * A policy gradient takes in game state observations and directly outputs a softmax for discreete
 * actions or sigmoids for continuous.
 *
 * To train it just use its own output as labels and multiplies them by the rewards
 * earned during that game/episode/trajectory. After a shitload of iterations good
 * outputs will slowly become more predominant and bad outputs will be suppressed.
 *
 * This formula below shows: good output = train output * rewards
 * J(θ)=∑ dπ(s) Vπ(s)
 *
 * Heres the update formula which shows a small change in the params will affect a small change in outcome
 * ∇θJ(θ)=Eπ[Qπ(s,a)∇θlnπθ(a|s)]
 * ∇θJ(θ)  : the small change in theta to move closer to better outcome as result of theta =
 * Eπ[]   : The expected reward based on the policy
 * Qπ(s,a) : This is where we multiply our reward (there are different versions)
 * ∇θlnπθ(a|s)    : This mess is a small change in the params
 *
 * This wont work right out of the box for 2 reasons
 *
 * 1) You need a way to decide how much reward to assign each step in a winning game
 *    This is completely arbitrary and this code sample chooses one of the common practices.
 *    But above you can see we scale our changes(∇θlnπθ(a|s)) by multiplying them by our reward(Qπ(s,a))
 *
 * 2) Because you cant init params to 0 or 1 your network will call certain probability
 *  distributions more often and assign them more rewards as a result. To compensate for this
 *  just divide params by their own weight. So a param thats 4x bigger will get 1/4 size updates to keep it
 *  from growing. That division would be shown as this:   ∇θπθ(a|s)/πθ(a|s)
 *  But that division isnt diff-able so we use the calculus chain rule, add that natural log,
 *  and change it to ∇θlnπθ(a|s). So ∇θlnπθ(a|s) = ∇θlnπθ(a|s)/πθ(a|s)
 *
 *  mind = blown. its that easy
 *
 * onger Version:
 *
 * Network Goals:
 * The vanilla policy gradient approximates a function of the best action to take based on a certain input known as state.
 * As written normally as πθ(a|s) The pi = policy/NN, theta = parameters and its a propability of action=a given state=s
 * VPG is stochastic meaning that even though certain actions have a high probability we are simply sampling from that
 * distribution. This means that our exploration will center more and more around the correct choice as our network is trained
 * stochastic sample done at line...   int action = new EnumeratedIntegerDistribution(actionSpace, dist.dup().data().asDouble()).sample();
 *
 *
 *
 * The first is a way to judge whether the action our policy returns fits our requirements. This is done via the Monte
 * Carlo method. Search "monte carlo pi approximation" for a simple intuitive explanation. Our strategy is we sample a number
 * of state action pairs and optimize for those that fit our definition of the function our VPG is trying to approximate.
 * In order to do this we complete an episode which gives us a total reward. Then assign that reward over each action taken
 * using the following formula.
 * Vπ(s)=Eπ[Gt|St=s]
 * Vπ(s)=Eπ The first part of formula shows the Value given policy of a state = Expected return given policy
 * [Gt | St=s] St=s equals current state and Gt is the discounted future rewards Gt=Rt+1+γRt+2+⋯
 * See method calcedValueOfState below to see how its implemented
 * Episodes where more good actions occur will have higher rewards and vice versa. So although actions will often receive
 * mixed signals, over many episodes those signals will sum to the correct values.
 *
 * The second thing we will need is a customized loss function
 *
 *
 *
 *
 * Some important notes there are many variation of these formulas and other optimizations so dont get confused with all the options
 * An episode could be a complete game or a single "good" action
 *
 */

public class VanillaPolicyGradient {

    private static String save = "D:\\Dropbox\\Apps\\RL\\";
    private static String name = "VPG_CartPole.dl4j";
    private static double gamma = .99; //discount rate
    int totalActions = 0;
    int stateSpace = 0;

    public static void main(String[] args)
    {
        VanillaPolicyGradient vpg = new VanillaPolicyGradient();

        ComputationGraph graph = null;

        try {

            graph = ComputationGraph.load(new File(save + name), true);

        }catch (Exception e){
            System.err.println("No existing file!");
        }

        graph =  vpg.train(1000000, graph);

        vpg.test(graph);

    }


    public ComputationGraph train(int epoch,  ComputationGraph graph)
    {

        NewGym newGym =null;
        try {
            newGym = new NewGym("CartPole-v1", NewGym.ObservationStyle.discreet, false);
        } catch (RuntimeException e) {
            System.err.println("Is gym_http_api server running?");
        }

        totalActions = newGym.actionSpace().getNumber();
        int[] actionSpace = new int[totalActions];
        for (int i = 0; i < actionSpace.length; i++) {
            actionSpace[i] = i;
        }
        NewGym.StepResponse stepResponse = newGym.reset();
        stateSpace = (int) stepResponse.getObservation().shape()[1];

        if (graph == null){
            graph = getVPGNetwork(stateSpace, totalActions);
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

            fit(graph, stateLs, actionLs, rewardLs);

            System.out.println("Reward: " + cumReward);
            //saveNet(graph);
        }

        return graph;
    }

    public void test(ComputationGraph graph)
    {

    }


    public void fit(ComputationGraph graph, ArrayList<INDArray> state, ArrayList<Integer> action, ArrayList<INDArray> reward){


        INDArray states = state.get(0);

        //calc discounted reward
        INDArray advantage = calcValueOfState(reward);

        //one hot actions
        INDArray actions = Nd4j.zeros(action.size(), totalActions);
        for (int i = 0; i < action.size(); i++) {
            actions.putScalar(i, action.get(i), 1);

            if (i != 0)
            {
                states = Nd4j.concat(0, states, state.get(i));
            }

        }

        DataSet ds = new DataSet();

        //INDArray temp = Nd4j.hstack( actions, advantage);
        INDArray temp = actions.mul(advantage);

        ds.setLabels(temp);
        ds.setFeatures(states);

        graph.fit(ds);


    }

    /**
     * In order to define how "good" an action is we associate it with its reward and future discounted
     *  Gt See into at top
     * As an optimization we will also do a baseline subtraction sutton & barto 2018 13.11
     * Our baseline will be an advantage Function which basically uses only the advantage
     * of taking the "correct" action over the incorrect.
     * A(s|a) = Q(s|a) - V(s) = index(i) - index(i+1) = total rewards inlcuding this action - total rewards without this action
     *
     *
     * @param reward
     * @return
     */
    public INDArray calcValueOfState(ArrayList<INDArray> reward)
    {
        ArrayList<INDArray> calcd = new ArrayList<>();
        INDArray out = Nd4j.zeros( reward.size(), 1);

        // gamma
        calcd.add(reward.get(reward.size()-1));

        for (int i = reward.size()-2; i >= 0; i--) {

            calcd.add(0, reward.get(i).add(calcd.get(0).mul(gamma)));


        }

        //baseline
        /*
        for (int i = 0; i < calcd.size()-1; i++) {

            out.put(i, calcd.get(i).sub(calcd.get(i+1)));

        }
        out.put(calcd.size()-1, calcd.get(calcd.size()-1));
         */

        for (int i = 0; i < calcd.size(); i++) {
            out.put(i, calcd.get(i));
        }

        out.subi(Nd4j.mean(out));
        out.divi(Nd4j.std(out));

        return out;
    }


    public ComputationGraph getVPGNetwork(int discObserv, int actions)
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

            confBuild.addLayer("action", new OutputLayer.Builder(new CustomLoss()).nOut(actions).activation(Activation.SOFTMAX).build(), "policy_2");


        confBuild
                .setOutputs("action")
                .setInputTypes(InputType.feedForward(discObserv))
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();

        //net.setListeners(new ScoreIterationListener(100));
        return net;



    }


    public void saveNet(ComputationGraph net){

        try {
            net.save(new File(save, name ));
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    /**
     *  L = -R(t) * lnpi(a|s)
     *
     * // TODO: 9/3/2020 this formula doesnt match what i did...
     *
     */
    class CustomLoss implements ILossFunction {


        @Override
        public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
            INDArray output = activationFn.getActivation(preOutput.dup(), true).addi(1e-5); //no zero for diff
            INDArray logOut = Transforms.log(output, true);

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
            INDArray loss = logOut.mul(labels); // -R(t) * logout

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
