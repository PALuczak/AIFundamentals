package pl.p.lodz.aifundamentals.searchalgorithm;

import java.util.ArrayList;
import java.util.Objects;
import java.util.stream.Collectors;

public class CanMis {
    public static void main(String[] args) {
        Search search = new Search(3,3);
        search.start();
        // queue uses breadth first search (also called optimal)
        // stack uses depth first search
        // parametrize
    }
}

class Search {

    private ArrayList<State> stateList = new ArrayList<State>();
    private ArrayList<State> history = new ArrayList<State>();
    final private State desiredState;

    Search(int cannibals, int missionaries) {
        stateList.add(new State(cannibals, 0, missionaries, 0, true, null));
        desiredState = new State(0, cannibals, 0, missionaries, false, null);
    }

    Search() {
        this(3,3);
    }

    void start() {
        State s;
        while (true) {
            if (stateList.isEmpty())
                throw new IllegalStateException("The problem has no solution");
            s = selectState();
            if (solutionFound(s)) {
                printReverseTree(s);
                break;
            }
            ArrayList<State> candidates = expandState(s);
            ArrayList<State> validStates = filterStates(candidates);
            updateStateList(validStates);
        }
    }

    private void printReverseTree(State s) {
        State node = s;
        while(node.parent != null){
            System.out.println(node);
            node = node.parent;
        }
        System.out.println(node);
    }

    private ArrayList<State> expandState(State s) {
        ArrayList<State> candidateList = new ArrayList<>();
        if (s.boatOnLeft) {
            // sending one cannibal to the right
            if (s.CL > 0)
                candidateList.add(new State(s.CL - 1, s.CR + 1, s.ML, s.MR, false, s));
            // sending one missionary to the right
            if (s.ML > 0)
                candidateList.add(new State(s.CL, s.CR, s.ML - 1, s.MR + 1, false, s));
            // sending two cannibals to the right
            if (s.CL >= 2)
                candidateList.add(new State(s.CL - 2, s.CR + 2, s.ML, s.MR, false, s));
            // sending two missionaries to the right
            if (s.ML >= 2)
                candidateList.add(new State(s.CL, s.CR, s.ML - 2, s.MR + 2, false, s));
            // sending one cannibal and one missionary to the right
            if (s.CL > 0 && s.ML > 0)
                candidateList.add(new State(s.CL - 1, s.CR + 1, s.ML - 1, s.MR + 1, false, s));
        } else {
            // sending one cannibal to the left
            if (s.CR > 0)
                candidateList.add(new State(s.CL + 1, s.CR - 1, s.ML, s.MR, true, s));
            // sending one missionary to the left
            if (s.MR > 0)
                candidateList.add(new State(s.CL, s.CR, s.ML + 1, s.MR - 1, true, s));
            // sending two cannibals to the left
            if (s.CR >= 2)
                candidateList.add(new State(s.CL + 2, s.CR - 2, s.ML, s.MR, true, s));
            // sending two missionaries to the left
            if (s.MR >= 2)
                candidateList.add(new State(s.CL, s.CR, s.ML + 2, s.MR - 2, true, s));
            // sending one cannibal and one missionary to the left
            if (s.CR > 0 && s.MR > 0)
                candidateList.add(new State(s.CL + 1, s.CR - 1, s.ML + 1, s.MR - 1, true, s));
        }
        return candidateList;
    }

    private void updateStateList(ArrayList<State> validStates) {
        stateList.addAll(validStates);
    }

    private ArrayList<State> filterStates(ArrayList<State> candidates) {
        return candidates.stream().filter((State val) -> isValid(val) && isNovel(val)).collect(Collectors.toCollection(ArrayList::new));
    }

    private boolean isValid(State val) {
        return (val.CL <= val.ML || val.ML == 0) && (val.CR <= val.MR || val.MR == 0);
    }

    private boolean isNovel(State val) {
        return !history.contains(val);
    }

    private boolean solutionFound(State s) {
        return s.equals(desiredState);
    }

    private State selectState() {
        State state = stateList.remove(0);
        history.add(state);
        return state;
    }
}

class State {
    int CL;
    int CR;
    int ML;
    int MR;
    boolean boatOnLeft;
    State parent;

    State(int cl, int cr, int ml, int mr, boolean left, State parentNode) {
        CL = cl;
        CR = cr;
        ML = ml;
        MR = mr;
        boatOnLeft = left;
        parent = parentNode;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        State state = (State) o;
        return CL == state.CL &&
                CR == state.CR &&
                ML == state.ML &&
                MR == state.MR &&
                boatOnLeft == state.boatOnLeft;
    }

    @Override
    public int hashCode() {
        return Objects.hash(CL, CR, ML, MR, boatOnLeft);
    }

    @Override
    public String toString() {
        return "State{" +
                "CL=" + CL +
                ", CR=" + CR +
                ", ML=" + ML +
                ", MR=" + MR +
                ", boatOnLeft=" + boatOnLeft +
                '}';
    }
}