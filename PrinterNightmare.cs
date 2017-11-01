using System;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

public class PrinterNightmare
{
    public Variable<int> numExamples;

    // primary RVs
    public VariableArray<bool> Fuse;
    public VariableArray<bool> Drum;
    public VariableArray<bool> Toner;
    public VariableArray<bool> Paper;
    public VariableArray<bool> Roller;
    public VariableArray<bool> Burning;
    public VariableArray<bool> Quality;
    public VariableArray<bool> Wrinkled;
    public VariableArray<bool> MultPages;
    public VariableArray<bool> PaperJam;

    // RVs representing parameters of the distrib. of the primary RVs
    public Variable<double> ProbFuse;
    public Variable<double> ProbDrum;
    public Variable<double> ProbToner;
    public Variable<double> ProbPaper;
    public Variable<double> ProbRoller;
    public VariableArray<double> CPTBurning;
    public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> CPTQuality;
    public VariableArray<VariableArray<double>, double[][]> CPTWrinkled;
    public VariableArray<VariableArray<double>, double[][]> CPTMultPages;
    public VariableArray<VariableArray<double>, double[][]> CPTPaperJam;

    // prior distrib. for the prob. and CPT variables
    public Variable<Beta> ProbFusePrior;
    public Variable<Beta> ProbDrumPrior;
    public Variable<Beta> ProbTonerPrior;
    public Variable<Beta> ProbPaperPrior;
    public Variable<Beta> ProbRollerPrior;
    public VariableArray<Beta> CPTBurningPrior;
    public VariableArray<VariableArray<VariableArray<Beta>, Beta[][]>, Beta[][][]> CPTQualityPrior;
    public VariableArray<VariableArray<Beta>, Beta[][]> CPTWrinkledPrior;
    public VariableArray<VariableArray<Beta>, Beta[][]> CPTMultPagesPrior;
    public VariableArray<VariableArray<Beta>, Beta[][]> CPTPaperJamPrior;

    // posterior distributions for the probability and CPT variables
    public Beta ProbFusePosterior;
    public Beta ProbDrumPosterior;
    public Beta ProbTonerPosterior;
    public Beta ProbPaperPosterior;
    public Beta ProbRollerPosterior;
    public Beta[] CPTBurningPosterior;
    public Beta[][][] CPTQualityPosterior;
    public Beta[][] CPTWrinkledPosterior;
    public Beta[][] CPTMultPagesPosterior;
    public Beta[][] CPTPaperJamPosterior;

    public InferenceEngine Engine = new InferenceEngine();

    public PrinterNightmare()
	{
        numExamples = Variable.New<int>();
        Range nRange = new Range(numExamples);

        Range fuseRange = new Range(2);
        Range drumRange = new Range(2);
        Range tonerRange = new Range(2);
        Range paperRange = new Range(2);
        Range rollerRange = new Range(2);
        Range burningRange = new Range(2);
        Range qualityRange = new Range(2);
        Range wrinkledRange = new Range(2);
        Range multPagesRange = new Range(2);
        Range paperJamRange = new Range(2);

        //
        // define priors and the parameters
        //

        ProbFusePrior = Variable.New<Beta>();
        ProbFuse = Variable<double>.Random(ProbFusePrior);

        ProbDrumPrior = Variable.New<Beta>();
        ProbDrum = Variable<double>.Random(ProbDrumPrior);

        ProbTonerPrior = Variable.New<Beta>();
        ProbToner = Variable<double>.Random(ProbTonerPrior);

        ProbPaperPrior = Variable.New<Beta>();
        ProbPaper = Variable<double>.Random(ProbPaperPrior);

        ProbRollerPrior = Variable.New<Beta>();
        ProbRoller = Variable<double>.Random(ProbRollerPrior);

        CPTBurningPrior = Variable.Array<Beta>(fuseRange);
        CPTBurning = Variable.Array<double>(fuseRange);
        CPTBurning[fuseRange] = Variable<double>.Random(CPTBurningPrior[fuseRange]);
        CPTBurning.SetValueRange(burningRange);

        CPTQualityPrior = Variable.Array(Variable.Array(Variable.Array<Beta>(drumRange), tonerRange), paperRange);
        CPTQuality = Variable.Array(Variable.Array(Variable.Array<double>(drumRange), tonerRange), paperRange);
        CPTQuality[paperRange][tonerRange][drumRange] = Variable<double>.Random(CPTQualityPrior[paperRange][tonerRange][drumRange]);
        CPTQuality.SetValueRange(qualityRange);

        CPTWrinkledPrior = Variable.Array(Variable.Array<Beta>(fuseRange), paperRange);
        CPTWrinkled = Variable.Array(Variable.Array<double>(fuseRange), paperRange);
        CPTWrinkled[paperRange][fuseRange] = Variable<double>.Random(CPTWrinkledPrior[paperRange][fuseRange]);
        CPTWrinkled.SetValueRange(wrinkledRange);

        CPTMultPagesPrior = Variable.Array(Variable.Array<Beta>(rollerRange), paperRange);
        CPTMultPages = Variable.Array(Variable.Array<double>(rollerRange), paperRange);
        CPTMultPages[paperRange][rollerRange] = Variable<double>.Random(CPTMultPagesPrior[paperRange][rollerRange]);
        CPTMultPages.SetValueRange(multPagesRange);

        CPTPaperJamPrior = Variable.Array(Variable.Array<Beta>(rollerRange), fuseRange);
        CPTPaperJam = Variable.Array(Variable.Array<double>(rollerRange), fuseRange);
        CPTPaperJam[fuseRange][rollerRange] = Variable<double>.Random(CPTPaperJamPrior[fuseRange][rollerRange]);
        CPTPaperJam.SetValueRange(paperJamRange);

        // define primary RVs
        Fuse = Variable.Array<bool>(nRange);
        Fuse[nRange] = Variable.Bernoulli(ProbFuse).ForEach(nRange);

        Drum = Variable.Array<bool>(nRange);
        Drum[nRange] = Variable.Bernoulli(ProbDrum).ForEach(nRange);

        Toner = Variable.Array<bool>(nRange);
        Toner[nRange] = Variable.Bernoulli(ProbToner).ForEach(nRange);

        Paper = Variable.Array<bool>(nRange);
        Paper[nRange] = Variable.Bernoulli(ProbPaper).ForEach(nRange);

        Roller = Variable.Array<bool>(nRange);
        Roller[nRange] = Variable.Bernoulli(ProbRoller).ForEach(nRange);

        Burning = AddChildFromOneParent(Fuse, CPTBurning);
        Quality = AddChildFromThreeParents(Drum, Toner, Paper, CPTQuality);
        Wrinkled = AddChildFromTwoParents(Fuse, Paper, CPTWrinkled);
        MultPages = AddChildFromTwoParents(Paper, Roller, CPTMultPages);
        PaperJam = AddChildFromTwoParents(Fuse, Roller, CPTPaperJam);
    }

    public void LearnParameters(
            bool[] fuse, bool[] drum, bool[] toner, bool[] paper, bool[] roller,
            bool[] burning, bool[] quality, bool[] wrinkled, bool[] multPages, bool[] paperJam
        )
    {
        // set number of examples at runtime;
        // assuming all data arrays are of the same length
        numExamples.ObservedValue = fuse.Length;

        // set data
        Fuse.ObservedValue = fuse;
        Drum.ObservedValue = drum;
        Toner.ObservedValue = toner;
        Paper.ObservedValue = paper;
        Roller.ObservedValue = roller;
        Burning.ObservedValue = burning;
        Quality.ObservedValue = quality;
        Wrinkled.ObservedValue = wrinkled;
        MultPages.ObservedValue = multPages;
        PaperJam.ObservedValue = paperJam;

        // inference
        ProbFusePosterior = Engine.Infer<Beta>(ProbFuse);
        ProbDrumPosterior = Engine.Infer<Beta>(ProbDrum);
        ProbTonerPosterior = Engine.Infer<Beta>(ProbToner);
        ProbPaperPosterior = Engine.Infer<Beta>(ProbPaper);
        ProbRollerPosterior = Engine.Infer<Beta>(ProbRoller);
        CPTBurningPosterior = Engine.Infer<Beta[]>(CPTBurning);
        CPTQualityPosterior = Engine.Infer<Beta[][][]>(CPTQuality);
        CPTWrinkledPosterior = Engine.Infer<Beta[][]>(CPTWrinkled);
        CPTMultPagesPosterior = Engine.Infer<Beta[][]>(CPTMultPages);
        CPTPaperJamPosterior = Engine.Infer<Beta[][]>(CPTPaperJam);
    }

    public static VariableArray<bool> AddChildFromOneParent(VariableArray<bool> parent, VariableArray<double> cpt)
    {
        var n = parent.Range;
        var child = Variable.Array<bool>(n);
        using (Variable.ForEach(n))
        {
            using (Variable.If(parent[n]))
                child[n] = Variable.Bernoulli(cpt[0]);
            using (Variable.IfNot(parent[n]))
                child[n] = Variable.Bernoulli(cpt[1]);
        }
        return child;
    }

    public static VariableArray<bool> AddChildFromTwoParents(VariableArray<bool> parent1, VariableArray<bool> parent2, VariableArray<VariableArray<double>, double[][]> cpt)
    {
        var n = parent1.Range;
        var child = Variable.Array<bool>(n);
        using (Variable.ForEach(n))
        {
            using (Variable.If(parent1[n]))
            {
                using (Variable.If(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[0][0]);
                using (Variable.IfNot(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[0][1]);
            }
            using (Variable.IfNot(parent1[n]))
            {
                using (Variable.If(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[1][0]);
                using (Variable.IfNot(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[1][1]);
            }
        }
        return child;
    }

    public static VariableArray<bool> AddChildFromThreeParents(VariableArray<bool> parent1, VariableArray<bool> parent2, VariableArray<bool> parent3, VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> cpt)
    {
        var n = parent1.Range;
        var child = Variable.Array<bool>(n);
        using (Variable.ForEach(n))
        {
            using (Variable.If(parent1[n]))
            {
                using (Variable.If(parent2[n]))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][0][0]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][0][1]);
                }
                using (Variable.IfNot(parent2[n]))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][1][0]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][1][1]);
                }
            }
            using (Variable.IfNot(parent1[n]))
            {
                using (Variable.If(parent2[n]))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][0][0]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][0][1]);
                }
                using (Variable.IfNot(parent2[n]))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][1][0]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][1][1]);
                }
            }
        }
        return child;
    }

    public static bool[][] GetData()
    {
        bool[][] data = new bool[][]
        {
            new bool[] { false, false, false, true, false, false, false, false, false, false, false, false, true, false, true },    // fuse assembly malfunction
            new bool[] { false, false, false, false, true, false, false, true, false, false, true, true, false, false, false },     // drum unit
            new bool[] { true, true, false, false, false, true, false, true, false, false, false, true, false, false, false },      // toner out
            new bool[] { true, false, true, false, true, false, true, false, true, true, false, true, true, false, false },         // poor paper quality
            new bool[] { false, false, false, false, false, false, true, false, false, false, false, false, false, true, true },    // worn roller
            new bool[] { false, false, false, true, false, false, false, false, false, false, false, false, true, false, false },   // burning smell
            new bool[] { true, true, true, false, true, true, false, true, false, false, true, true, false, false, false },         // poor print quality
            new bool[] { false, false, true, false, false, false, false, false, true, false, false, false, true, true, true },      // wrinkled pages
            new bool[] { false, false, true, false, false, false, true, false, true, false, false, false, false, false, true },     // multiple pages fed
            new bool[] { false, false, true, true, false, false, true, true, true, true, false, false, false, true, true }          // paper jam
        };
        return data;
    }

    public static void Main()
    {
        Rand.Restart(2017);

        PrinterNightmare model = new PrinterNightmare();
        if (model.Engine.Algorithm is GibbsSampling)
        {
            Console.WriteLine("This example does not run with Gibbs Sampling.");
            return;
        }

        bool[][] data = GetData();

        model.LearnParameters(data[0], data[1], data[2], data[3], data[4],
                              data[5], data[6], data[7], data[8], data[9]);
    }
}
